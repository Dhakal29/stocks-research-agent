"""Exercise Google Search grounding using the real SDK and mocked HTTP."""

import asyncio
import copy
import json

import httpx
import pytest

from nepse_agent.agent import EvidenceError, NepseResearchAgent, ResearchError, parse_gemini_report


@pytest.fixture(autouse=True)
def isolate_local_book_retrieval(monkeypatch):
    # Provider transport tests must not download models or depend on local PDFs.
    from types import SimpleNamespace
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "0")
    monkeypatch.setattr(
        "nepse_agent.rag_engine.get_rag_store",
        lambda: SimpleNamespace(retrieve_for_analysis=lambda *args, **kwargs: []),
    )


def grounded_response():
    first = "नबिल परीक्षण विवरण।"
    second = "Synthetic company news from another publisher."
    return {
        "candidates": [{
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [
                {"text": "Private fixture reasoning", "thought": True},
                {"text": first}, {"text": second},
            ]},
            "groundingMetadata": {
                "webSearchQueries": ["NABIL site:merolagani.com", "NABIL site:sharesansar.com"],
                "groundingChunks": [
                    {"web": {"title": "MeroLagani", "uri": "https://www.merolagani.com/CompanyDetail.aspx?symbol=NABIL"}},
                    {"web": {"title": "ShareSansar", "uri": "https://www.sharesansar.com/company/NABIL"}},
                    {"web": {"title": "Uncited source", "uri": "https://example.com/uncited"}},
                ],
                "groundingSupports": [
                    {"segment": {"partIndex": 1, "startIndex": 0, "endIndex": len(first.encode("utf-8"))}, "groundingChunkIndices": [0]},
                    {"segment": {"partIndex": 2, "startIndex": 0, "endIndex": len(second.encode("utf-8"))}, "groundingChunkIndices": [1]},
                ],
                "searchEntryPoint": {"renderedContent": "<div>Google Search suggestions fixture</div>"},
            },
        }],
    }


def google_agent(responder, **kwargs):
    return NepseResearchAgent("test-key", "gemini-3.5-flash-lite", httpx.MockTransport(responder), provider="gemini", **kwargs)


def test_gemini_registers_search_tool_and_preserves_query_and_provenance():
    calls = []

    def respond(request):
        calls.append(request)
        assert request.method == "POST"
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = json.loads(request.content)
        assert payload["tools"] == [{"googleSearch": {}}]
        assert payload["contents"][0]["parts"][0]["text"] == "Analyze NABIL recent news"
        assert "site:sharesansar.com" in payload["systemInstruction"]["parts"][0]["text"]
        return httpx.Response(200, json=grounded_response())

    report = asyncio.run(google_agent(respond).research("Analyze NABIL recent news"))
    assert len(calls) == 1
    assert "नबिल परीक्षण विवरण। [MeroLagani]" in report.markdown
    assert "Private fixture reasoning" not in report.markdown
    assert report.source_domains == ["merolagani.com", "sharesansar.com"]
    assert len(report.cited_sources) == 2
    assert len(report.searched_urls) == 3
    assert len(report.search_queries) == 2
    assert report.search_suggestions_html == "<div>Google Search suggestions fixture</div>"


def test_narrow_coverage_triggers_one_targeted_search_then_succeeds():
    prompts = []

    def respond(request):
        prompts.append(json.loads(request.content)["systemInstruction"]["parts"][0]["text"])
        data = grounded_response()
        if len(prompts) == 1:
            data["candidates"][0]["groundingMetadata"]["groundingSupports"] = data["candidates"][0]["groundingMetadata"]["groundingSupports"][:1]
        return httpx.Response(200, json=data)

    report = asyncio.run(google_agent(respond).research("NABIL"))
    assert len(prompts) == 2
    assert "previous attempt cited only these publisher sites: merolagani.com" in prompts[1]
    assert len(report.source_domains) == 2


def test_two_urls_on_one_publisher_and_uncited_sites_are_insufficient():
    calls = []

    def respond(request):
        calls.append(request)
        data = grounded_response()
        data["candidates"][0]["groundingMetadata"]["groundingChunks"][1]["web"]["uri"] = "https://eng.merolagani.com/NewsDetail.aspx?newsID=fixture"
        return httpx.Response(200, json=data)

    with pytest.raises(EvidenceError, match="after two attempts"):
        asyncio.run(google_agent(respond).research("NABIL"))
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["no_search", "no_citations", "invalid_utf8_offset", "unsafe_url"])
def test_unusable_grounding_is_rejected(failure):
    data = grounded_response()
    metadata = data["candidates"][0]["groundingMetadata"]
    if failure == "no_search":
        metadata["webSearchQueries"] = []
    elif failure == "no_citations":
        metadata["groundingSupports"] = []
    elif failure == "invalid_utf8_offset":
        metadata["groundingSupports"] = [copy.deepcopy(metadata["groundingSupports"][0])]
        metadata["groundingSupports"][0]["segment"]["endIndex"] = 1
    else:
        for chunk in metadata["groundingChunks"]:
            chunk["web"]["uri"] = "javascript:fixture"
    with pytest.raises(EvidenceError):
        parse_gemini_report(data, "NABIL", "fixture-time", "fixture-model")


def test_google_redirect_domains_come_from_actual_destinations():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            data = grounded_response()
            chunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
            for index in [0, 1]:
                chunks[index]["web"]["title"] = f"Article {index}"
                chunks[index]["web"]["uri"] = f"https://vertexaisearch.cloud.google.com/grounding-api-redirect/fixture{index}"
            return httpx.Response(200, json=data)
        assert "x-goog-api-key" not in request.headers
        assert request.url.host == "vertexaisearch.cloud.google.com"
        target = "https://www.merolagani.com/news" if request.url.path.endswith("0") else "https://www.sharesansar.com/news"
        return httpx.Response(302, headers={"Location": target})

    report = asyncio.run(google_agent(respond).research("NABIL"))
    assert len(requests) == 3
    assert report.source_domains == ["merolagani.com", "sharesansar.com"]
    assert all("vertexaisearch.cloud.google.com" in source.url for source in report.cited_sources)


def test_google_redirects_with_domain_titles_do_not_require_extra_requests():
    def respond(request):
        assert request.method == "POST"
        data = grounded_response()
        chunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
        for index, domain in enumerate(["merolagani.com", "sharesansar.com"]):
            chunks[index]["web"].update(title=domain, uri=f"https://vertexaisearch.cloud.google.com/grounding-api-redirect/fixture{index}")
        return httpx.Response(200, json=data)

    assert len(asyncio.run(google_agent(respond).research("NABIL")).source_domains) == 2


def test_gemini_api_error_does_not_retry_or_expose_key():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"code": 429, "message": "private-body test-key", "status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(ResearchError, match="HTTP 429") as error:
        asyncio.run(google_agent(respond).research("NABIL"))
    assert len(calls) == 1
    assert "private-body" not in str(error.value)
    assert "test-key" not in str(error.value)


def test_provider_and_site_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai")
    monkeypatch.delenv("NEPSE_PROVIDER", raising=False)
    assert NepseResearchAgent().provider == "gemini"
    assert NepseResearchAgent(api_key="explicit-openai").provider == "openai"
    monkeypatch.setenv("NEPSE_PROVIDER", "openai")
    assert NepseResearchAgent().provider == "openai"
    assert NepseResearchAgent(min_sites=3).min_sites == 3
    with pytest.raises(ResearchError, match="between 2 and 10"):
        NepseResearchAgent(min_sites=1)
    with pytest.raises(ResearchError, match="between 2 and 10"):
        monkeypatch.setenv("NEPSE_MIN_SITES", "invalid")
        NepseResearchAgent()


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_gemini_transient_errors_retry_then_recover_without_changing_query(status, monkeypatch, caplog):
    from unittest.mock import AsyncMock
    calls = []
    sleep = AsyncMock()
    monkeypatch.setattr("nepse_agent.agent.asyncio.sleep", sleep)
    monkeypatch.setattr("nepse_agent.agent.random.uniform", lambda *args: 0.0)
    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) < 3:
            return httpx.Response(status, json={"error": {
                "code": status, "message": "private-body test-key", "status": "UNAVAILABLE",
            }})
        return httpx.Response(200, json=grounded_response())
    with caplog.at_level("WARNING", logger="nepse_agent.agent"):
        report = asyncio.run(google_agent(respond).research("unhpl"))
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert calls[0]["contents"][0]["parts"][0]["text"] == "unhpl"
    assert report.query == "unhpl"
    assert report.source_domains == ["merolagani.com", "sharesansar.com"]
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]
    assert f"[Gemini retry] HTTP {status}" in caplog.text
    assert "private-body" not in caplog.text
    assert "test-key" not in caplog.text


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_gemini_persistent_transient_errors_stop_after_three_attempts(status, monkeypatch):
    from unittest.mock import AsyncMock
    calls = []
    sleep = AsyncMock()
    monkeypatch.setattr("nepse_agent.agent.asyncio.sleep", sleep)
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {
            "code": status, "message": "private-body test-key", "status": "UNAVAILABLE",
        }})
    with pytest.raises(ResearchError, match=f"temporarily unavailable.*HTTP {status}.*3 attempts") as error:
        asyncio.run(google_agent(respond).research("unhpl"))
    assert len(calls) == 3
    assert sleep.await_count == 2
    assert "GEMINI_API_KEY" not in str(error.value)
    assert "private-body" not in str(error.value)
    assert "test-key" not in str(error.value)


@pytest.mark.parametrize("status,hint", [
    (400, "request settings"), (401, "GEMINI_API_KEY"),
    (403, "permissions"), (404, "GEMINI_MODEL"),
])
def test_gemini_configuration_errors_fail_immediately(status, hint, monkeypatch):
    from unittest.mock import AsyncMock
    sleep = AsyncMock()
    monkeypatch.setattr("nepse_agent.agent.asyncio.sleep", sleep)
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": status, "message": "private-body test-key"}})
    with pytest.raises(ResearchError, match=hint):
        asyncio.run(google_agent(respond).research("UNHPL"))
    assert len(calls) == 1
    sleep.assert_not_awaited()


def test_gemini_generation_deadline_cancels_slow_request(monkeypatch):
    monkeypatch.setattr("nepse_agent.agent.GEMINI_REQUEST_TIMEOUT", 0.02)
    calls, cancelled = [], []
    async def respond(request):
        calls.append(request)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    with pytest.raises(ResearchError, match="including retries"):
        asyncio.run(google_agent(respond).research("unhpl"))
    assert len(calls) == 1
    assert cancelled == [True]


def test_gemini_503_in_assessment_retries_only_that_stage(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from nepse_agent.rag_engine import DocumentChunk
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    monkeypatch.setattr("nepse_agent.agent.asyncio.sleep", AsyncMock())
    chunk = DocumentChunk("actual-id", "Book.pdf", "Cash flow", 9, "Operating cash flow should support earnings.", 0.8)
    retrievals = []
    def retrieve(query, evidence="", top_k=8):
        retrievals.append((query, evidence))
        return [chunk]
    monkeypatch.setattr(
        "nepse_agent.rag_engine.get_rag_store",
        lambda: SimpleNamespace(retrieve_for_analysis=retrieve),
    )
    prompts = []
    def respond(request):
        prompts.append(json.loads(request.content)["systemInstruction"]["parts"][0]["text"])
        if len(prompts) == 2:
            return httpx.Response(503, json={"error": {"code": 503, "message": "temporary overload", "status": "UNAVAILABLE"}})
        data = grounded_response()
        if len(prompts) == 3:
            candidate = data["candidates"][0]
            candidate["content"]["parts"][-1]["text"] += " [BOOK:actual-id]"
            candidate["groundingMetadata"]["groundingSupports"][-1]["segment"]["endIndex"] = len(
                candidate["content"]["parts"][-1]["text"].encode("utf-8")
            )
        return httpx.Response(200, json=data)
    report = asyncio.run(google_agent(respond).research("unhpl"))
    assert len(prompts) == 3
    assert prompts[1] == prompts[2]
    assert "web-evidence stage" in prompts[0]
    assert "WEB RESEARCH RESPONSE" in prompts[1]
    assert len(retrievals) == 1
    assert "[Book.pdf | Cash flow | PDF page 9]" in report.markdown
