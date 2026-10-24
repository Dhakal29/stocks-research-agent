"""Dated quotes from the anonymous public API used by nepalstock.com."""

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from .trend_analysis import evaluate_stock_trend

logger = logging.getLogger(__name__)
NEPSE_BASE_URL = "https://www.nepalstock.com"
NEPAL_TIMEZONE = ZoneInfo("Asia/Kathmandu")
QUOTE_TIMEOUT = 20


class QuoteError(ValueError):
    """An unavailable or inconsistent official quote; never contains session data."""


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    company_name: str
    price: float
    business_date: str
    last_updated_at: str | None
    price_type: str
    freshness: str
    fifty_two_week_low: float | None
    fifty_two_week_high: float | None
    source_url: str
    data_url: str


@dataclass(frozen=True)
class QuoteResult:
    symbol: str
    quote: MarketQuote | None = None
    unavailable_reason: str = ""


def requested_symbols(query: str, researched_at: str) -> list[str]:
    """Find ticker requests without inserting live data into historical reports."""
    query = query.strip()
    cutoff = datetime.fromisoformat(researched_at).astimezone(NEPAL_TIMEZONE)
    if re.search(
        r"\b(historical|history|yesterday|last (?:week|month|year)|\d+ days? ago)\b", query, re.IGNORECASE,
    ):
        return []
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if any(value != cutoff.date().isoformat() for value in dates):
        return []
    if any(int(year) != cutoff.year for year in re.findall(r"\b(?:19|20)\d{2}\b", query)):
        return []
    named_dates = re.findall(
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2})"
        r"(?:,?\s+\d{4})?\b", query, re.IGNORECASE,
    )
    for value in named_dates:
        parsed = None
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                parsed = datetime.strptime(f"{value} {cutoff.year}", fmt).date()
                break
            except ValueError:
                continue
        if parsed != cutoff.date():
            return []
    if (re.search(r"\bas (?:of|at)\s+(?!today\b|now\b|(?:the )?current\b)", query, re.IGNORECASE)
            and not dates and not named_dates):
        return []
    if re.search(r"\b(?:USA|NYSE|NASDAQ|American|Indian|NSE|BSE|US (?:stocks?|markets?)|United States)\b", query, re.IGNORECASE):
        return []
    bare_symbol = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,11}", query))
    if not bare_symbol and not re.search(
        r"\b(prices?|quotes?|CMP|LTP|analy[sz]e|analysis|fundamentals?|valuation|invest(?:ment|ing)?|"
        r"compare|reports?)\b", query, re.IGNORECASE,
    ):
        return []
    ignored = {
        "NEPSE", "NPR", "EPS", "ROE", "ROA", "PE", "PB", "CMP", "LTP", "RAG", "PDF", "AGM",
        "BS", "AD", "FY", "Q1", "Q2", "Q3", "Q4", "NEWS", "HELP", "HELLO", "TODAY", "CURRENT",
        "PRICE", "QUOTE", "STOCK", "MARKET", "ANALYSIS", "ANALYZE", "RESEARCH", "COMPARE", "AND",
        "WHAT", "IS", "THE", "FOR", "OF", "PLEASE", "LATEST", "VALUE", "AT", "ON", "IN", "AS", "DATA", "FULL",
    }
    candidates = re.findall(r"\b[A-Z][A-Z0-9]{1,11}\b", query)
    if bare_symbol:
        candidates.append(query.upper())
    # Also support ordinary lowercase requests such as "analyze unhpl" or "unhpl price".
    candidates.extend(value.upper() for value in re.findall(
        r"\b(?:analy[sz]e|research|compare|ticker|symbol|and|versus|vs\.?|with|price of|quote for)\s+"
        r"([a-z][a-z0-9]{1,11})\b", query, re.IGNORECASE,
    ))
    candidates.extend(value.upper() for value in re.findall(
        r"\b([a-z][a-z0-9]{1,11})\s+(?:stock|price|quote|fundamentals|valuation)\b", query, re.IGNORECASE,
    ))
    return [value for value in dict.fromkeys(candidates) if value not in ignored]


def _public_session_token(session: dict, helper: bytes) -> str:
    """Follow the public website's WASM session flow, without user credentials."""
    from wasmtime import Config, Engine, Instance, Module, Store

    if len(helper) > 1024 * 1024:
        raise QuoteError("NEPSE's public session helper has an unexpected format.")
    config = Config()
    config.consume_fuel = True
    engine = Engine(config)
    store = Store(engine)
    store.set_fuel(100_000)
    store.set_limits(memory_size=32 * 1024 * 1024)
    # No WASI, filesystem, network or other host capabilities are supplied.
    exports = Instance(store, Module(engine, helper), []).exports(store)
    salts = [session[f"salt{index}"] for index in range(1, 6)]
    if any(type(value) is not int for value in salts):
        raise QuoteError("NEPSE's public session has an unexpected format.")
    swapped = [salts[0], salts[1], salts[3], salts[2], salts[4]]
    offsets = [exports[name](store, *(salts if index == 0 else swapped))
               for index, name in enumerate(("cdx", "rdx", "bdx", "ndx", "mdx"))]
    token = session["accessToken"]
    if not isinstance(token, str) or len(set(offsets)) != 5 or offsets != sorted(offsets):
        raise QuoteError("NEPSE's public session has an unexpected format.")
    if any(not 0 <= offset < len(token) for offset in offsets):
        raise QuoteError("NEPSE's public session has an unexpected format.")
    return "".join(char for index, char in enumerate(token) if index not in offsets)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise QuoteError("NEPSE returned an invalid quote timestamp.") from None
    return parsed.replace(tzinfo=NEPAL_TIMEZONE) if parsed.tzinfo is None else parsed.astimezone(NEPAL_TIMEZONE)


def parse_quote(data: dict, symbol: str, security_id: int, market: dict, researched_at: str) -> MarketQuote:
    security, values = data.get("securityData"), data.get("securityMcsData")
    if not isinstance(security, dict) or not isinstance(values, dict):
        raise QuoteError("NEPSE returned no quote for this security.")
    if (security.get("symbol") != symbol or str(security.get("id")) != str(security_id)
            or str(values.get("securityId")) != str(security_id)):
        raise QuoteError("NEPSE's quote does not match the requested symbol and company ID.")
    cutoff = datetime.fromisoformat(researched_at).astimezone(NEPAL_TIMEZONE)
    try:
        business_date = date.fromisoformat(values.get("businessDate", ""))
    except (TypeError, ValueError):
        raise QuoteError("NEPSE returned no valid trading date.") from None
    updated = _timestamp(values.get("lastUpdatedDateTime"))
    if business_date > cutoff.date() or (updated and updated > cutoff):
        raise QuoteError("NEPSE's quote is dated after the research cutoff.")
    if updated and updated.date() != business_date:
        raise QuoteError("NEPSE's trading date and quote timestamp disagree.")
    last_price = _number(values.get("lastTradedPrice"))
    if last_price is None:
        raise QuoteError("NEPSE returned no valid last traded price.")
    # A date alone never proves that an LTP is a closing price.
    try:
        market_as_of = _timestamp(market.get("asOf"))
    except QuoteError:
        market_as_of = None
    same_session = bool(market_as_of and market_as_of.date() == business_date and market_as_of <= cutoff)
    close = _number(values.get("closePrice"))
    price, price_type = last_price, "Last traded price (live/delay status unavailable)"
    if same_session and market.get("isOpen") == "CLOSE" and close is not None:
        price, price_type = close, "Closing price"
    elif same_session and market.get("isOpen") in {"OPEN", "HALT"}:
        price_type = "Intraday last traded price (delay status unavailable)"
    low, high = _number(values.get("fiftyTwoWeekLow")), _number(values.get("fiftyTwoWeekHigh"))
    if low is not None and high is not None and low > high:
        low = high = None
    return MarketQuote(
        symbol, security.get("securityName") or symbol, price, business_date.isoformat(),
        updated.isoformat() if updated else None, price_type,
        "current_date" if business_date == cutoff.date() else "older_session", low, high,
        f"{NEPSE_BASE_URL}/company/detail/{security_id}", f"{NEPSE_BASE_URL}/api/nots/security/{security_id}",
    )


async def fetch_market_quotes(
    query: str, researched_at: str, *, transport: httpx.AsyncBaseTransport | None = None,
) -> list[QuoteResult]:
    symbols = requested_symbols(query, researched_at)
    if not symbols:
        return []
    try:
        async with asyncio.timeout(QUOTE_TIMEOUT), httpx.AsyncClient(
            base_url=NEPSE_BASE_URL, timeout=8, transport=transport,
            headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache", "Accept": "application/json"},
        ) as http:
            async def get(path: str, **kwargs):
                response = await http.get(path, **kwargs)
                response.raise_for_status()
                return response

            session = (await get("/api/authenticate/prove", params={"_": str(time.time_ns())})).json()
            helper = (await get("/assets/prod/css.wasm")).content
            token = await asyncio.to_thread(_public_session_token, session, helper)
            http.headers["Authorization"] = "Salter " + token
            securities = (await get("/api/nots/security", params={"nonDelisted": "true"})).json()
            if not isinstance(securities, list):
                raise QuoteError("NEPSE returned an invalid securities list.")
            by_symbol = {item.get("symbol"): item for item in securities if isinstance(item, dict)}
            symbols = [symbol for symbol in symbols if symbol in by_symbol][:10]
            if not symbols:
                return []  # Not a listed NEPSE ticker; leave other markets to web research.
            try:
                market = (await get("/api/nots/nepse-data/market-open")).json()
                market = market if isinstance(market, dict) else {}
            except (httpx.HTTPError, ValueError):
                market = {}  # A price can still be verified without guessing its price type.

            async def lookup(symbol: str) -> QuoteResult:
                try:
                    security_id = int(by_symbol[symbol]["id"])
                    data = (await get(f"/api/nots/security/{security_id}")).json()
                    if not isinstance(data, dict):
                        raise QuoteError("NEPSE returned an invalid company quote.")
                    return QuoteResult(symbol, parse_quote(data, symbol, security_id, market, researched_at))
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    reason = str(exc) if isinstance(exc, QuoteError) else "The official NEPSE quote could not be retrieved."
                    return QuoteResult(symbol, unavailable_reason=reason)

            results = await asyncio.gather(*(lookup(symbol) for symbol in symbols))
            for result in results:
                logger.info("[NEPSE quote] %s: %s", result.symbol,
                            f"NPR {result.quote.price:.2f}, trading date {result.quote.business_date}, {result.quote.freshness}"
                            if result.quote else result.unavailable_reason)
            return results
    except Exception as exc:
        # Session tokens and provider bodies must not appear in logs or reports.
        reason = str(exc) if isinstance(exc, QuoteError) else "The official NEPSE quote service is unavailable."
        logger.warning("[NEPSE quote] Official quote lookup unavailable (%s).", type(exc).__name__)
        return [QuoteResult(symbol, unavailable_reason=reason) for symbol in symbols]


def quote_context(results: list[QuoteResult]) -> str:
    if not results:
        return ""
    return (
        "\n--- OFFICIAL NEPSE QUOTES: APPLICATION DATA ---\n"
        + json.dumps([asdict(result) for result in results], ensure_ascii=False)
        + "\nThese quotes were fetched directly from nepalstock.com for this request. "
        "Use them for price-dependent calculations; never replace them with an older search snippet. "
        "The application renders the price, quote date, last-update time, price type and 52-week range separately. "
        "Omit those quote fields from your response and Executive Snapshot. Do not repeat a quote table. "
        "The supplied quote does not replace web research: use the internet search tool for company identity, "
        "financial statements, fundamentals and dated news requested by the user, and cite the retrieved evidence. "
        "An older_session quote is only the latest available official quote, not today's verified price. "
        "For an unavailable quote, state the valuation limitation and do not supply a current price from another site. "
        "Do not infer a closing price or market holiday from the age of a quote. "
        "Do not fill a missing 52-week bound from another date.\n"
        "--- END OFFICIAL NEPSE QUOTES ---\n"
    )


def quote_markdown(results: list[QuoteResult]) -> str:
    sections = []
    for result in results:
        quote = result.quote
        if quote is None:
            sections.append(f"**{result.symbol} current market price: unavailable.** {result.unavailable_reason}")
            continue
        label = "Current market price" if quote.freshness == "current_date" else "Latest available price (older session)"
        updated = datetime.fromisoformat(quote.last_updated_at).strftime("%H:%M:%S") if quote.last_updated_at else "Time unavailable"
        low = f"NPR {quote.fifty_two_week_low:,.2f}" if quote.fifty_two_week_low is not None else "unavailable"
        high = f"NPR {quote.fifty_two_week_high:,.2f}" if quote.fifty_two_week_high is not None else "unavailable"
        freshness = "Quote dated today" if quote.freshness == "current_date" else "Older session; current-session price unverified"
        trend_res = evaluate_stock_trend(
            current_price=quote.price,
            fifty_two_week_low=quote.fifty_two_week_low,
            fifty_two_week_high=quote.fifty_two_week_high,
        )
        pos_52w = trend_res.indicators.get("52w_position_percentile", "N/A")
        sections.append(
            f"**{quote.symbol} — {quote.company_name}**\n\n"
            "| Field | Official NEPSE data |\n| --- | --- |\n"
            f"| {label} | NPR {quote.price:,.2f} |\n"
            f"| Quote date | {quote.business_date} |\n"
            f"| Source last update (Asia/Kathmandu, UTC+05:45) | {updated} |\n"
            f"| Price type | {quote.price_type} |\n"
            f"| 52-week range | {low} – {high} |\n"
            f"| 52-week position | {pos_52w} from annual low |\n"
            f"| Technical trend (Chukul model) | **{trend_res.trend}** ({trend_res.score}/{trend_res.max_score}) |\n"
            f"| Trend summary | {trend_res.summary} |\n"
            f"| Freshness | {freshness} |\n"
            f"| Source | [Nepal Stock Exchange]({quote.source_url}) |"
        )
    return "## Official NEPSE market quote\n\n" + "\n\n".join(sections)
