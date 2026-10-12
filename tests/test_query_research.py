"""Non-symbol questions through the real provider adapters and A2A transport.

Provider replies are synthetic; these tests do not verify live market figures.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from nepse_agent.agent import NepseResearchAgent
from nepse_agent.client import ask_agent_report
from nepse_agent.server import create_app
from nepse_agent.market_data import (
    QuoteError, QuoteResult, fetch_market_quotes, parse_quote, quote_markdown, requested_symbols,
)


@pytest.fixture(autouse=True)
def isolate_local_book_retrieval(monkeypatch):
    from unittest.mock import AsyncMock
    monkeypatch.setattr("nepse_agent.agent.fetch_market_quotes", AsyncMock(return_value=[]))
    # Provider transport tests must not download models or depend on local PDFs.
    from types import SimpleNamespace
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "0")
    monkeypatch.setattr(
        "nepse_agent.rag_engine.get_rag_store",
        lambda: SimpleNamespace(retrieve_for_analysis=lambda *args, **kwargs: []),
    )


SOURCES = [
    "https://www.nrb.org.np/monetary-policy/",
    "https://kathmandupost.com/money",
]


def response_for(provider, query, answer, sources=SOURCES):
    if provider == "gemini":
        return {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [{"text": answer}]},
                "groundingMetadata": {
                    "webSearchQueries": [query],
                    "groundingChunks": [{"web": {"uri": url, "title": "Synthetic source"}} for url in sources],
                    "groundingSupports": [{
                        "segment": {"startIndex": 0, "endIndex": len(answer.encode("utf-8"))},
                        "groundingChunkIndices": list(range(len(sources))),
                    }],
                },
            }],
        }
    return {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call", "status": "completed",
                "action": {"queries": [query], "sources": [{"url": url} for url in sources]},
            },
            {
                "type": "message",
                "content": [{
                    "type": "output_text", "text": answer,
                    "annotations": [{
                        "type": "url_citation", "url": url, "title": "Synthetic source",
                        "start_index": 0, "end_index": len(answer),
                    } for url in sources],
                }],
            },
        ],
    }


def analysis_response_for(provider, answer):
    if provider == "gemini":
        return {"candidates": [{
            "finishReason": "STOP", "content": {"role": "model", "parts": [{"text": answer}]},
        }]}
    return {"status": "completed", "output": [{
        "type": "message", "content": [{"type": "output_text", "text": answer, "annotations": []}],
    }]}


def request_content(provider, request, *, search=True):
    payload = json.loads(request.content)
    if provider == "gemini":
        if search:
            assert payload["tools"] == [{"googleSearch": {}}]
        else:
            assert not payload.get("tools")
        return payload["contents"][0]["parts"][0]["text"], payload["systemInstruction"]["parts"][0]["text"]
    if search:
        assert payload["tool_choice"] == "required"
        assert payload["tools"] == [{"type": "web_search", "external_web_access": True}]
    else:
        assert "tool_choice" not in payload
        assert not payload.get("tools")
    return payload["input"], payload["instructions"]


@pytest.mark.parametrize("provider", ["gemini", "openai"])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("answer,hint", [
    ("Unknown source [WEB:999] [WEB:1] [WEB:2]", "unknown or malformed web source IDs"),
    ("Invented source [fake](https://invented.example/quote) [WEB:1] [WEB:2]", "raw web links"),
    ("Uncited company facts", "did not cite the retrieved web evidence"),
    ("Only one publisher [WEB:1]", "enough distinct publisher sites"),
    ("Unretrieved book [WEB:1] [WEB:2] [BOOK:999]", "book passages that were not retrieved"),
    ("Malformed book [WEB:1] [WEB:2] [BOOK:", "malformed book citation"),
])
def test_analysis_citation_retry_does_not_repeat_search_or_retrieval(provider, recover, answer, hint, monkeypatch):
    from types import SimpleNamespace
    from nepse_agent.agent import AnalysisEvidenceError
    from nepse_agent.rag_engine import DocumentChunk
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    chunk = DocumentChunk("cash", "Book.pdf", "Cash Flow", 7,
                          "Compare operating cash flow with net income to assess the quality of earnings.", 0.8)
    retrievals = []
    def retrieve(query, evidence="", top_k=8):
        retrievals.append((query, evidence))
        return [chunk]
    monkeypatch.setattr("nepse_agent.rag_engine.get_rag_store", lambda: SimpleNamespace(retrieve_for_analysis=retrieve))
    prompts = []

    def respond(request):
        user_input, instructions = request_content(provider, request, search=not prompts)
        assert user_input == "UNHPL"
        prompts.append(instructions)
        if len(prompts) == 1:
            return httpx.Response(200, json=response_for(provider, user_input, "Verified company evidence."))
        text = "Assessment [WEB:1] [WEB:2]" if recover and len(prompts) == 3 else answer
        return httpx.Response(200, json=analysis_response_for(provider, text + " [BOOK:1]"))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    if recover:
        report = asyncio.run(agent.research("UNHPL"))
        assert report.source_domains == ["kathmandupost.com", "nrb.org.np"]
        assert "[Book.pdf | Cash Flow | PDF page 7]" in report.markdown
        assert report.book_sources[0]["chunk_id"] == "cash"
        assert report.book_sources[0]["citation_id"] == "1"
        assert "[WEB:" not in report.markdown
    else:
        with pytest.raises(AnalysisEvidenceError, match=hint):
            asyncio.run(agent.research("UNHPL"))
    assert len(prompts) == 3  # One web request, two synthesis attempts.
    assert len(retrievals) == 1
    assert "previous response failed evidence validation" in prompts[-1]
    assert "WEB RESEARCH RESPONSE" in prompts[1]
    assert "Allowed book citation markers for this request: [BOOK:1]" in prompts[-1]
    if "[BOOK:999]" in answer:
        assert "Invalid book IDs: '999'" in prompts[-1]


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_synthesis_only_counts_cited_sources_and_preserves_search_provenance(provider, monkeypatch):
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    sources = SOURCES + ["https://www.sebon.gov.np/issuer"]
    calls = []

    def respond(request):
        user_input, instructions = request_content(provider, request, search=not calls)
        calls.append(instructions)
        if len(calls) == 1:
            return httpx.Response(200, json=response_for(provider, user_input, "Verified stock evidence.", sources))
        assert "[WEB:3]" in instructions
        return httpx.Response(200, json=analysis_response_for(provider, "Assessment from two retrieved sources [WEB:1] [WEB:3]."))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    report = asyncio.run(agent.research("UNHPL"))
    assert len(calls) == 2
    assert report.source_domains == ["nrb.org.np", "sebon.gov.np"]
    assert {source.url for source in report.cited_sources} == {sources[0], sources[2]}
    assert set(report.searched_urls) == set(sources)
    assert report.search_queries == ["UNHPL"]


@pytest.fixture
def research_clock(monkeypatch):
    class Clock(datetime):
        current = datetime(2026, 10, 8, 18, 20, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            assert tz == ZoneInfo("Asia/Kathmandu")
            return cls.current.astimezone(tz)

    monkeypatch.setattr("nepse_agent.agent.datetime", Clock)
    return Clock


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_current_stock_clock_refreshes_per_request_in_nepal(provider, research_clock, caplog):
    query = "UNHPL"
    prompts = []

    def respond(request):
        user_input, instructions = request_content(provider, request)
        assert user_input == query
        prompts.append(instructions)
        return httpx.Response(200, json=response_for(provider, query, "Synthetic dated stock evidence."))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    reports = []
    with caplog.at_level("INFO", logger="nepse_agent"):
        for instant in [
            datetime(2026, 10, 8, 18, 20, tzinfo=timezone.utc),
            datetime(2026, 10, 9, 18, 20, tzinfo=timezone.utc),
        ]:
            research_clock.current = instant
            reports.append(asyncio.run(agent.research(query)))

    expected = [
        ("2026-10-09T00:05:00+05:45", "2026-10-09", "2026-09-09T00:05:00+05:45", "2026-10-08T18:20:00+00:00"),
        ("2026-10-10T00:05:00+05:45", "2026-10-10", "2026-09-10T00:05:00+05:45", "2026-10-09T18:20:00+00:00"),
    ]
    assert len(prompts) == len(reports) == 2
    for instructions, report, (timestamp, day, news_start, utc) in zip(prompts, reports, expected):
        assert instructions.startswith("--- APPLICATION DATE AND TIME: AUTHORITATIVE ---")
        assert f"Current research time: {timestamp} (Asia/Kathmandu)." in instructions
        assert f"Current date in Nepal (Gregorian/AD): {day}." in instructions
        assert "Current local time: 00:05:00 +0545" in instructions
        assert f"Same instant in UTC: {utc}." in instructions
        assert f"Default stock search date: {day}." in instructions
        assert f"{news_start} through {timestamp}" in instructions
        assert f"the ticker/company and {day} in initial web searches" in instructions
        assert report.researched_at == timestamp
        assert f"Researched at: {timestamp}" in report.markdown
        assert f"[research clock] Current date/time: {timestamp} (Asia/Kathmandu)" in caplog.text


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_web_first_and_coverage_retry_share_the_request_cutoff(provider, research_clock, monkeypatch):
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    research_clock.current = datetime(2026, 10, 9, 23, 59, 50, tzinfo=ZoneInfo("Asia/Kathmandu"))
    query = "UNHPL"
    prompts = []

    def respond(request):
        user_input, instructions = request_content(provider, request, search=len(prompts) < 2)
        assert user_input == query
        prompts.append(instructions)
        research_clock.current += timedelta(minutes=1)
        if len(prompts) == 3:
            return httpx.Response(200, json=analysis_response_for(provider, "Assessment [WEB:1] [WEB:2]."))
        sources = SOURCES[:1] if len(prompts) == 1 else SOURCES
        return httpx.Response(200, json=response_for(provider, query, "Synthetic verified stock evidence.", sources))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    report = asyncio.run(agent.research(query))
    assert len(prompts) == 3
    assert "web-evidence stage" in prompts[0]
    assert "Keep answering the original question" in prompts[1]
    assert "WEB RESEARCH RESPONSE" in prompts[2]
    for instructions in prompts:
        assert instructions.startswith("--- APPLICATION DATE AND TIME: AUTHORITATIVE ---")
        assert "Current research time: 2026-10-09T23:59:50+05:45" in instructions
        assert "Default stock search date: 2026-10-09." in instructions
    assert research_clock.current.date().isoformat() == "2026-10-10"
    assert report.researched_at == "2026-10-09T23:59:50+05:45"


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_historical_stock_request_retains_its_requested_period(provider, research_clock):
    query = "Research NABIL as of 2024-01-15, using only information published by then."
    answer = "Historical NABIL evidence as of 2024-01-15; older financial periods are labeled."

    def respond(request):
        user_input, instructions = request_content(provider, request)
        assert user_input == query
        assert "Current date in Nepal (Gregorian/AD): 2026-10-09." in instructions
        assert "For an explicit historical request, use the user's requested period instead of the default date/window." in instructions
        return httpx.Response(200, json=response_for(provider, query, answer))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    report = asyncio.run(agent.research(query))
    assert report.query == query
    assert answer in report.markdown
    assert report.researched_at == "2026-10-09T00:05:00+05:45"


@pytest.mark.parametrize("provider", ["gemini", "openai"])
@pytest.mark.parametrize("query,answer", [
    ("Give me today's market summary in 200 words", "## Market summary\nSynthetic market session evidence."),
    ("Latest Nepal economic news in Nepali", "## आर्थिक समाचार\nपरीक्षण समाचार।"),
    ("Compare NABIL and EBL's latest quarterly results", "## Comparison\nSynthetic comparable-period evidence."),
    ("Give me today's US market summary", "## US market summary\nSynthetic US session evidence."),
    ("How do interest rates affect share prices?", "## Interest rates\nSynthetic explanation with evidence."),
])
def test_questions_preserve_scope_and_accept_sources_beyond_stock_portals(provider, query, answer):
    calls = []

    def respond(request):
        calls.append(request)
        user_input, instructions = request_content(provider, request)
        assert user_input == query
        assert "A stock symbol\nis optional" in instructions
        assert "Choose the answer structure from the question" in instructions
        assert "Asia/Kathmandu" in instructions
        return httpx.Response(200, json=response_for(provider, query, answer))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    report = asyncio.run(agent.research(query))
    assert len(calls) == 1
    assert report.query == query
    assert answer in report.markdown
    assert report.source_domains == ["kathmandupost.com", "nrb.org.np"]
    assert report.search_queries == [query]
    assert {source.url for source in report.cited_sources} == set(SOURCES)


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_broader_search_retry_keeps_original_market_question(provider):
    query = "Give me today's NEPSE market summary"
    prompts = []

    def respond(request):
        user_input, instructions = request_content(provider, request)
        assert user_input == query
        prompts.append(instructions)
        sources = SOURCES[:1] if len(prompts) == 1 else SOURCES
        return httpx.Response(200, json=response_for(provider, query, "Synthetic market summary.", sources))

    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    report = asyncio.run(agent.research(query))
    assert len(prompts) == 2
    assert "Keep answering the original question and its requested timeframe/format" in prompts[1]
    assert report.source_domains == ["kathmandupost.com", "nrb.org.np"]


def test_market_summary_over_a2a_has_query_provenance_and_market_skill():
    async def scenario():
        query = "Give me today's market summary"
        answer = "## Market summary\nSynthetic latest verified session; not a live quote."

        def respond(request):
            user_input, _ = request_content("openai", request)
            assert user_input == query
            return httpx.Response(200, json=response_for("openai", query, answer))

        agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider="openai")
        app = create_app(agent, "http://testserver")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            card = (await client.get("/.well-known/agent-card.json")).json()
            market_skill = next(skill for skill in card["skills"] if skill["id"] == "market_and_web_research")
            assert query in market_skill["examples"]
            report = await ask_agent_report(query, "http://testserver", client)
            assert report["query"] == query
            assert answer in report["markdown"]
            assert report["source_domains"] == ["kathmandupost.com", "nrb.org.np"]
            assert report["search_queries"] == [query]

    asyncio.run(scenario())


OFFICIAL_CUTOFF = "2026-10-12T16:00:00+05:45"


def official_quote_data():
    # Synthetic provider data in the exchange's verified public response format.
    return {
        "securityData": {"id": 2831, "symbol": "UNHPL", "securityName": "Union Hydropower Limited"},
        "securityMcsData": {
            "securityId": "2831", "lastTradedPrice": 352.9, "closePrice": 352.9,
            "businessDate": "2026-10-12", "lastUpdatedDateTime": "2026-10-12T14:59:58.324608",
            "fiftyTwoWeekLow": 337, "fiftyTwoWeekHigh": 571,
        },
    }


def official_market():
    return {"isOpen": "CLOSE", "asOf": "2026-10-12T15:00:00", "id": 80}


def official_transport(calls, *, quote_data=None, fail=False):
    from wasmtime import wat2wasm
    # Real WASM execution verifies both salt argument ordering and token slicing.
    functions = " ".join(
        f'(func (export "{name}") (param i32 i32 i32 i32 i32) (result i32) local.get {index})'
        for index, name in enumerate(("cdx", "rdx", "bdx", "ndx", "mdx"))
    )
    helper = bytes(wat2wasm("(module " + functions + ")"))

    def respond(request):
        calls.append(request)
        assert request.url.host == "www.nepalstock.com"
        assert request.method == "GET"
        assert "x-goog-api-key" not in request.headers
        if request.url.path == "/api/authenticate/prove":
            assert request.url.params["_"]
            assert "Authorization" not in request.headers
            if fail:
                return httpx.Response(503, text="private-session-body")
            return httpx.Response(200, json={
                "accessToken": "a0b1c2d3e4f", "salt1": 1, "salt2": 3, "salt3": 7, "salt4": 5, "salt5": 9,
            })
        if request.url.path == "/assets/prod/css.wasm":
            return httpx.Response(200, content=helper)
        assert request.headers["Authorization"] == "Salter abcdef"
        if request.url.path == "/api/nots/security":
            assert request.url.params["nonDelisted"] == "true"
            return httpx.Response(200, json=[{"id": 2831, "symbol": "UNHPL"}])
        if request.url.path == "/api/nots/nepse-data/market-open":
            return httpx.Response(200, json=official_market())
        assert request.url.path == "/api/nots/security/2831"
        return httpx.Response(200, json=quote_data if quote_data is not None else official_quote_data())

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("provider", ["gemini", "openai"])
@pytest.mark.parametrize("web_first", [False, True])
def test_official_october_12_quote_survives_search_and_analysis(provider, web_first, research_clock, monkeypatch):
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1" if web_first else "0")
    monkeypatch.setattr("nepse_agent.agent.fetch_market_quotes", fetch_market_quotes)
    research_clock.current = datetime.fromisoformat(OFFICIAL_CUTOFF)
    official_calls, prompts = [], []

    def respond(request):
        user_input, instructions = request_content(provider, request, search=not prompts)
        assert user_input == "UNHPL"
        assert '"price": 352.9' in instructions
        assert '"business_date": "2026-10-12"' in instructions
        assert "never replace them with an older search snippet" in instructions
        assert "a0b1c2d3e4f" not in instructions
        prompts.append(instructions)
        if len(prompts) == 1:
            return httpx.Response(200, json=response_for(provider, "UNHPL", "Verified company fundamentals."))
        return httpx.Response(200, json=analysis_response_for(provider, "Company assessment [WEB:1] [WEB:2]."))

    agent = NepseResearchAgent(
        "fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider,
        market_transport=official_transport(official_calls),
    )
    report = asyncio.run(agent.research("UNHPL"))
    assert len(prompts) == (2 if web_first else 1)
    assert sum(call.url.path == "/api/authenticate/prove" for call in official_calls) == 1
    assert sum(call.url.path == "/api/nots/security/2831" for call in official_calls) == 1
    assert report.markdown.count("## Official NEPSE market quote") == 1
    assert "| Current market price | NPR 352.90 |" in report.markdown
    assert "| Quote date | 2026-10-12 |" in report.markdown
    assert "| 14:59:58 |" in report.markdown
    assert "| Price type | Closing price |" in report.markdown
    assert "NPR 337.00 – NPR 571.00" in report.markdown
    assert "NPR 354.00" not in report.markdown
    assert report.market_quotes[0]["quote"]["price"] == 352.9
    assert report.market_quotes[0]["quote"]["freshness"] == "current_date"
    assert "nepalstock.com" in report.source_domains
    assert any(source.url.endswith("/company/detail/2831") for source in report.cited_sources)


@pytest.mark.parametrize("provider", ["gemini", "openai"])
@pytest.mark.parametrize("missing_search_first", [False, True])
def test_stale_generated_price_is_normalized_without_a_research_retry(provider, missing_search_first, research_clock, monkeypatch):
    monkeypatch.setattr("nepse_agent.agent.fetch_market_quotes", fetch_market_quotes)
    research_clock.current = datetime.fromisoformat(OFFICIAL_CUTOFF)
    official_calls, prompts = [], []

    def respond(request):
        user_input, instructions = request_content(provider, request)
        prompts.append(instructions)
        text = (
            "**Current Market Price:** NPR 354.00\nQuote Date: October 8, 2026\nTimestamp: Time unavailable\n\n"
            "Verified company fundamentals; valuation uses the supplied official quote."
        )
        data = response_for(provider, user_input, text)
        if missing_search_first and len(prompts) == 1:
            if provider == "gemini":
                data["candidates"][0]["groundingMetadata"] = {}
            else:
                data["output"][0]["status"] = "failed"
        return httpx.Response(200, json=data)

    agent = NepseResearchAgent(
        "fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider,
        market_transport=official_transport(official_calls),
    )
    report = asyncio.run(agent.research("UNHPL"))
    assert "NPR 352.90" in report.markdown
    assert "354.00" not in report.markdown
    assert "October 8" not in report.markdown
    assert "Verified company fundamentals" in report.markdown
    assert report.markdown.count("| Current market price |") == 1
    assert len(prompts) == (2 if missing_search_first else 1)
    if missing_search_first:
        assert "previous response failed evidence validation" in prompts[1]
    assert sum(call.url.path == "/api/nots/security/2831" for call in official_calls) == 1


def test_unavailable_exchange_does_not_promote_a_search_price(research_clock, monkeypatch):
    monkeypatch.setattr("nepse_agent.agent.fetch_market_quotes", fetch_market_quotes)
    research_clock.current = datetime.fromisoformat(OFFICIAL_CUTOFF)
    calls = []

    def respond(request):
        user_input, instructions = request_content("openai", request)
        assert '"quote": null' in instructions
        assert "do not supply a current price from another site" in instructions
        return httpx.Response(200, json=response_for("openai", user_input, "Current valuation cannot be verified."))

    agent = NepseResearchAgent(
        "fixture-key", "fixture-model", httpx.MockTransport(respond),
        market_transport=official_transport(calls, fail=True),
    )
    report = asyncio.run(agent.research("UNHPL"))
    assert "UNHPL current market price: unavailable" in report.markdown
    assert report.market_quotes[0]["quote"] is None
    assert "nepalstock.com" not in report.source_domains
    assert "private-session-body" not in report.markdown


def test_old_official_quote_is_labeled_and_not_assumed_to_be_closing():
    data = official_quote_data()
    data["securityMcsData"].update(
        businessDate="2026-10-08", lastUpdatedDateTime="2026-10-08T14:59:00", lastTradedPrice=354, closePrice=354,
    )
    quote = parse_quote(data, "UNHPL", 2831, official_market(), OFFICIAL_CUTOFF)
    markdown = quote_markdown([QuoteResult("UNHPL", quote)])
    assert quote.freshness == "older_session"
    assert "Closing" not in quote.price_type
    assert "Latest available price (older session)" in markdown
    assert "current-session price unverified" in markdown
    assert "| Current market price |" not in markdown
    assert "2026-10-08" in markdown


def test_missing_quote_time_and_partial_range_remain_explicit():
    data = official_quote_data()
    data["securityMcsData"].update(lastUpdatedDateTime=None, fiftyTwoWeekLow=340.1, fiftyTwoWeekHigh=None)
    quote = parse_quote(data, "UNHPL", 2831, {}, OFFICIAL_CUTOFF)
    markdown = quote_markdown([QuoteResult("UNHPL", quote)])
    assert quote.last_updated_at is None
    assert "Time unavailable" in markdown
    assert "NPR 340.10 – unavailable" in markdown
    assert "Closing" not in quote.price_type
    assert "16:00:00" not in markdown


def test_market_open_does_not_turn_a_populated_close_field_into_closing_price():
    market = {**official_market(), "isOpen": "OPEN"}
    quote = parse_quote(official_quote_data(), "UNHPL", 2831, market, OFFICIAL_CUTOFF)
    assert quote.price_type.startswith("Intraday last traded price")


@pytest.mark.parametrize("field,value,hint", [
    ("lastTradedPrice", None, "valid last traded price"),
    ("lastTradedPrice", float("nan"), "valid last traded price"),
    ("lastTradedPrice", -354, "valid last traded price"),
    ("lastTradedPrice", True, "valid last traded price"),
    ("businessDate", None, "valid trading date"),
    ("businessDate", "2026-10-13", "after the research cutoff"),
    ("lastUpdatedDateTime", "2026-10-12T17:00:00", "after the research cutoff"),
    ("lastUpdatedDateTime", "2026-10-11T14:00:00", "disagree"),
    ("lastUpdatedDateTime", "not a timestamp", "invalid quote timestamp"),
    ("securityId", "2769", "does not match"),
])
def test_invalid_exchange_prices_and_dates_are_rejected(field, value, hint):
    data = official_quote_data()
    data["securityMcsData"][field] = value
    with pytest.raises(QuoteError, match=hint):
        parse_quote(data, "UNHPL", 2831, official_market(), OFFICIAL_CUTOFF)


def test_company_id_cannot_substitute_another_hydropower_company():
    data = official_quote_data()
    data["securityData"]["symbol"] = "NHDL"
    with pytest.raises(QuoteError, match="does not match"):
        parse_quote(data, "UNHPL", 2831, official_market(), OFFICIAL_CUTOFF)


@pytest.mark.parametrize("query,symbols", [
    ("unhpl", ["UNHPL"]),
    ("Analyze UNHPL", ["UNHPL"]),
    ("analyze unhpl", ["UNHPL"]),
    ("unhpl price", ["UNHPL"]),
    ("Compare UNHPL and NABIL", ["UNHPL", "NABIL"]),
    ("Research UNHPL as of 2024-01-15", []),
    ("Analyze UNHPL in 2024", []),
    ("UNHPL price as of 2026-10-12", ["UNHPL"]),
    ("UNHPL price on October 12, 2026", ["UNHPL"]),
    ("UNHPL price on October 8, 2026", []),
    ("Tell us the UNHPL price", ["UNHPL"]),
    ("UNHPL on 2026-10-08", []),
    ("UNHPL yesterday", []),
    ("Latest UNHPL news", []),
    ("Give me today's market summary", []),
    ("How do interest rates affect share prices?", []),
    ("Give me today's US market summary", []),
])
def test_official_quote_lookup_preserves_query_scope(query, symbols):
    assert requested_symbols(query, OFFICIAL_CUTOFF) == symbols


def test_historical_request_makes_no_current_quote_http_calls():
    def must_not_call(request):
        pytest.fail("A historical request must not fetch a current quote")
    results = asyncio.run(fetch_market_quotes(
        "Analyze UNHPL as of 2024-01-15", OFFICIAL_CUTOFF, transport=httpx.MockTransport(must_not_call),
    ))
    assert results == []


@pytest.mark.parametrize("field", [
    "**Current Market Price:** NPR 354.00", "Current Price: NPR 354.00", "Price: NPR 354.00",
    "| Market Price (NPR) | 354.00 |", "Quote Date: October 8, 2026", "52-Week Range: NPR 340.10",
])
def test_generated_quote_field_variants_cannot_override_application_data(field):
    from nepse_agent.agent import _normalize_provider_quote_fields
    normalized = _normalize_provider_quote_fields(field + "\n\nVerified company fundamentals.")
    assert normalized == "Verified company fundamentals."


@pytest.mark.parametrize("provider", ["gemini", "openai"])
@pytest.mark.parametrize("price", ["352.90", "354.00"])
def test_repeated_quote_fields_in_synthesis_do_not_restart_any_stage(provider, price, research_clock, monkeypatch):
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    monkeypatch.setattr("nepse_agent.agent.fetch_market_quotes", fetch_market_quotes)
    research_clock.current = datetime.fromisoformat(OFFICIAL_CUTOFF)
    official_calls, prompts = [], []

    def respond(request):
        user_input, instructions = request_content(provider, request, search=not prompts)
        prompts.append(instructions)
        if len(prompts) == 1:
            return httpx.Response(200, json=response_for(provider, user_input, "Verified company fundamentals."))
        return httpx.Response(200, json=analysis_response_for(provider,
            f"| Field | Value |\n| --- | --- |\n| Current Market Price | NPR {price} |\n"
            "| Quote Date | October 8, 2026 |\n| Sector | Hydropower [WEB:1] |\n\n"
            "Company assessment [WEB:1] [WEB:2]."
        ))

    agent = NepseResearchAgent(
        "fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider,
        market_transport=official_transport(official_calls),
    )
    report = asyncio.run(agent.research("unhpl"))
    assert len(prompts) == 2
    assert sum(call.url.path == "/api/nots/security/2831" for call in official_calls) == 1
    assert report.markdown.count("| Current market price | NPR 352.90 |") == 1
    assert "354.00" not in report.markdown
    assert "October 8" not in report.markdown
    assert "Hydropower" in report.markdown
    assert "Company assessment" in report.markdown


@pytest.mark.parametrize("markdown,expected", [
    (
        "| Field | Value |\n| --- | --- |\n| Current Market Price | NPR 354 |\n"
        "| EPS | 1.80 [WEB:1] |\n| 52-week range | 340.10 – unavailable |",
        "| Field | Value |\n| --- | --- |\n| EPS | 1.80 [WEB:1] |",
    ),
    (
        "| Symbol | Price (NPR) | EPS | Quote date |\n| --- | --- | --- | --- |\n"
        "| UNHPL | 354 | 1.80 [WEB:1] | 2026-10-08 |",
        "| Symbol | EPS |\n| --- | --- |\n| UNHPL | 1.80 [WEB:1] |",
    ),
    (
        "### Current Market Price\nNPR 354.00\n\n### Fundamentals\nEPS 1.80 [WEB:1].",
        "### Fundamentals\nEPS 1.80 [WEB:1].",
    ),
    (
        "- **Current Market Price:**\n  NPR 354.00\n- **EPS:** 1.80 [WEB:1]",
        "- **EPS:** 1.80 [WEB:1]",
    ),
])
def test_quote_normalization_preserves_fundamentals_and_citations(markdown, expected):
    from nepse_agent.agent import _normalize_provider_quote_fields
    assert _normalize_provider_quote_fields(markdown) == expected


def test_removed_quote_citations_do_not_inflate_source_coverage():
    from nepse_agent.agent import Source, _attach_official_quotes, _build_report, _source_link
    old = Source("Old quote", SOURCES[0], "nrb.org.np")
    facts = Source("Company facts", SOURCES[1], "kathmandupost.com")
    evidence = _build_report("unhpl", OFFICIAL_CUTOFF, "fixture", [
        f"Current Market Price: NPR 354.00 {_source_link(old)}\n\nCompany evidence {_source_link(facts)}"
    ], {old.url: old, facts.url: facts}, [old.url, facts.url], ["unhpl"])
    quote = parse_quote(official_quote_data(), "UNHPL", 2831, official_market(), OFFICIAL_CUTOFF)
    report = _attach_official_quotes(evidence, [QuoteResult("UNHPL", quote)])
    assert report.source_domains == ["kathmandupost.com", "nepalstock.com"]
    assert old.url not in {source.url for source in report.cited_sources}
    assert old.url in report.searched_urls
