"""Financial benchmarking module.

Searches the web for the most recent financial statements of a company and its
sector peers, then structures revenue growth, gross margins, R&D spend, and
cash conversion cycles into a clean comparative Markdown table.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

RESEARCH_TIMEZONE = ZoneInfo("Asia/Kathmandu")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BENCHMARK_SYSTEM_PROMPT = """
You are a financial benchmarking analyst. Your only job for this request is to
search the internet for the most recent publicly available financial statements
(annual or quarterly) of the requested company and its closest sector peers,
then present the data in a single, clean comparative Markdown table.

REQUIRED METRICS (columns per company):
1. Revenue Growth (YoY %)
2. Gross Margin (%)
3. R&D Spend (as % of revenue, or absolute amount if % is unavailable)
4. Cash Conversion Cycle (CCC) – Days Sales Outstanding + Days Inventory
   Outstanding − Days Payable Outstanding. If full CCC data is unavailable,
   report the components you can verify.

RULES:
- Search for the latest annual report, quarterly earnings release, or
  investor-relations disclosure available online.
- Compare the focal company against 2-4 meaningful peers in the same sector.
  For NEPSE/Nepal companies, try to find local peers first; if unavailable,
  use the closest international comparables and label them clearly.
- Present all figures for the same reporting period where possible; if periods
  differ, note the period beside each company name.
- Use a Markdown table as the primary output. Add a short paragraph (≤ 5
  sentences) below the table summarising key takeaways.
- Cite every figure with an inline source link immediately after the number.
- If a metric cannot be verified from any public source, write "N/A (unverified)"
  in that cell — never guess or interpolate.
- Do NOT produce a full company report, investment verdict, or scorecard.
  Focus solely on the four benchmark metrics in the table.
""".strip()

BENCHMARK_USER_TEMPLATE = (
    "Search the internet for the most recent financial statements of {company} "
    "and its closest 2-4 sector peers. Find revenue growth (YoY %), gross margin (%), "
    "R&D spend (% of revenue), and cash conversion cycle (days) from the latest "
    "annual report or earnings release. Today is {today}."
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    company: str
    researched_at: str
    markdown: str
    elapsed_seconds: float = 0.0
    cited_sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BenchmarkingAgent:
    """Thin wrapper around NepseResearchAgent tuned for benchmarking queries."""

    def __init__(self) -> None:
        # Reuse the same env-based config the main agent uses.
        from nepse_agent.agent import NepseResearchAgent
        self._agent = NepseResearchAgent()
        # Benchmarking needs fewer peer sites than full company research.
        self._agent.min_sites = 1

    async def benchmark(self, company: str) -> BenchmarkReport:
        """Run a web-search benchmarking pass for *company* and return results."""
        import time
        from nepse_agent.agent import (
            RESEARCH_TIMEZONE,
            ResearchError,
            validate_query,
        )

        company = company.strip()
        if not company:
            raise ResearchError("Provide a company name or ticker to benchmark.")

        today = datetime.now(RESEARCH_TIMEZONE).strftime("%Y-%m-%d")
        researched_at = datetime.now(RESEARCH_TIMEZONE).isoformat(timespec="seconds")

        # SHORT query → forces Gemini to actually run a Google Search rather than
        # answer from memory (a long pre-filled template suppresses the search call).
        query = BENCHMARK_USER_TEMPLATE.format(company=company, today=today)

        # The detailed table format and rules live in the system instruction only.
        instructions = (
            f"Current date (Asia/Kathmandu): {researched_at}\n\n"
            + BENCHMARK_SYSTEM_PROMPT
            + "\n\n"
            "After searching, present results as a Markdown table:\n\n"
            "| Company (Period) | Revenue Growth YoY (%) | Gross Margin (%) | R&D Spend (% rev) | CCC (days) |\n"
            "|---|---|---|---|---|\n\n"
            "Fill every cell with a verified figure and an inline citation. "
            "Below the table, add ≤5 sentences summarising key takeaways.\n"
            "Use the internet search tool. Cite at least 2 distinct publisher sites.\n"
        )

        start = time.perf_counter()
        logger.info("[benchmarking] Starting benchmark for %r", company)

        # Route through the correct provider search method.
        search = (
            self._agent._research_gemini
            if self._agent.provider == "gemini"
            else self._agent._research_openai
        )

        report = await search(query, researched_at, instructions=instructions)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info("[benchmarking] Completed in %.2fs for %r", elapsed, company)

        cited = [{"title": s.title, "url": s.url} for s in report.cited_sources]
        return BenchmarkReport(
            company=company,
            researched_at=researched_at,
            markdown=report.markdown,
            elapsed_seconds=elapsed,
            cited_sources=cited,
        )


def run_benchmark(company: str) -> BenchmarkReport:
    """Synchronous convenience wrapper (used from Streamlit)."""
    agent = BenchmarkingAgent()
    return asyncio.run(agent.benchmark(company))
