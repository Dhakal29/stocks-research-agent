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


@pytest.fixture(autouse=True)
def isolate_local_book_retrieval(monkeypatch):
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
