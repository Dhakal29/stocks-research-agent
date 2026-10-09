"""Provider fixtures are synthetic; these tests make no external API calls."""

import asyncio
import copy
import json
import runpy

import httpx
import pytest
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from google.protobuf.json_format import MessageToDict

from nepse_agent.agent import NepseResearchAgent, ResearchError, parse_report
from nepse_agent.client import ask_agent, ask_agent_report
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


SOURCE_URL = "https://www.merolagani.com/CompanyDetail.aspx?symbol=NABIL"
RESEARCH_TIME = "2026-10-04T12:00:00+05:45"


def provider_response():
    marker = "citefixture1"
    text = "Synthetic NABIL test report. " + marker
    return {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "queries": ["NABIL news", "NABIL financial results"], "sources": [
                    {"url": SOURCE_URL}, {"url": SOURCE_URL},
                    {"url": "https://www.sharesansar.com/company/NABIL"},
                ]},
            },
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [{
                        "type": "url_citation", "title": "Company source", "url": SOURCE_URL,
                        "start_index": text.index(marker), "end_index": len(text),
                    }, {
                        "type": "url_citation", "title": "ShareSansar", "url": "https://www.sharesansar.com/company/NABIL",
                        "start_index": text.index(marker), "end_index": len(text),
                    }],
                }],
            },
        ],
    }


def test_research_requires_live_search_and_preserves_provenance():
    def respond(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["input"] == "NABIL"
        assert payload["model"] == "test-model"
        assert payload["tools"] == [{"type": "web_search", "external_web_access": True}]
        assert payload["tool_choice"] == "required"
        assert payload["include"] == ["web_search_call.action.sources"]
        assert payload["store"] is False
        assert "Asia/Kathmandu" in payload["instructions"]
        return httpx.Response(200, json=provider_response())

    agent = NepseResearchAgent("test-key", "test-model", httpx.MockTransport(respond))
    report = asyncio.run(agent.research(" NABIL "))
    assert "Synthetic NABIL test report." in report.markdown
    assert f"[Company source]({SOURCE_URL})" in report.markdown
    assert "cite" not in report.markdown
    assert len(report.cited_sources) == 2
    assert len(report.searched_urls) == 2
    assert report.source_domains == ["merolagani.com", "sharesansar.com"]
    assert report.search_queries == ["NABIL news", "NABIL financial results"]
    assert report.researched_at.endswith("+05:45")


def test_citation_spans_retain_claims_and_group_multiple_sources():
    data = provider_response()
    part = data["output"][1]["content"][0]
    part["text"] = "A sourced claim must survive rendering."
    part["annotations"] = part["annotations"][:1]
    first = part["annotations"][0]
    first.update(start_index=0, end_index=len(part["text"]), title="A [report]")
    second = copy.deepcopy(first)
    second.update(url="https://example.com/report(1)", title="Second report")
    part["annotations"].extend([second, {"type": "url_citation", "url": "javascript:alert(1)"}])
    report = parse_report(data, "NABIL", RESEARCH_TIME, "test-model")
    assert part["text"] in report.markdown
    assert "A \\[report\\]" in report.markdown
    assert "https://example.com/report%281%29" in report.markdown
    assert "javascript:" not in report.markdown
    assert len(report.cited_sources) == 2


@pytest.mark.parametrize("failure", ["no_search", "no_citations", "incomplete", "no_text"])
def test_unverified_or_incomplete_reports_are_rejected(failure):
    data = provider_response()
    if failure == "no_search":
        data["output"].pop(0)
    elif failure == "no_citations":
        data["output"][1]["content"][0]["annotations"] = []
    elif failure == "incomplete":
        data["status"] = "incomplete"
    else:
        data["output"][1]["content"][0]["text"] = ""
    with pytest.raises(ResearchError):
        parse_report(data, "NABIL", RESEARCH_TIME, "test-model")


@pytest.mark.parametrize("status,hint", [(401, "OPENAI_API_KEY"), (429, "quota"), (500, "later")])
def test_api_errors_are_actionable_without_exposing_provider_body(status, hint):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"error": "private-body"}))
    with pytest.raises(ResearchError, match=hint) as error:
        asyncio.run(NepseResearchAgent("test-key", transport=transport).research("NABIL"))
    assert "private-body" not in str(error.value)
    assert "test-key" not in str(error.value)


def test_transport_and_invalid_json_errors():
    def timeout(request):
        raise httpx.ReadTimeout("fixture", request=request)

    with pytest.raises(ResearchError, match="timed out"):
        asyncio.run(NepseResearchAgent("test-key", transport=httpx.MockTransport(timeout)).research("NABIL"))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))
    with pytest.raises(ResearchError, match="invalid JSON"):
        asyncio.run(NepseResearchAgent("test-key", transport=transport).research("NABIL"))


def test_configuration_and_invalid_queries_do_not_call_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ResearchError, match="Set OPENAI_API_KEY"):
        NepseResearchAgent(provider="openai")

    def must_not_call(request):
        pytest.fail("Invalid queries must not trigger an API call")

    agent = NepseResearchAgent("test-key", transport=httpx.MockTransport(must_not_call))
    for query in ["  ", "a" * 2001]:
        with pytest.raises(ValueError):
            asyncio.run(agent.research(query))


def test_a2a_discovery_client_and_report_artifacts():
    async def scenario():
        provider = httpx.MockTransport(lambda request: httpx.Response(200, json=provider_response()))
        app = create_app(NepseResearchAgent("test-key", transport=provider), "http://testserver")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as http:
            card = (await http.get("/.well-known/agent-card.json")).json()
            assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
            assert card["skills"][0]["id"] == "nepse_company_research"
            request = SendMessageRequest(message=new_text_message("NABIL", role=Role.ROLE_USER))
            response = await http.post("/", headers={"A2A-Version": "1.0"}, json={
                "jsonrpc": "2.0", "id": "test", "method": "SendMessage", "params": MessageToDict(request),
            })
            task = response.json()["result"]["task"]
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            assert [artifact["name"] for artifact in task["artifacts"]] == [
                "nepse-research-report", "nepse-research-data",
            ]
            assert task["artifacts"][1]["parts"][0]["data"]["cited_sources"][0]["url"] == SOURCE_URL
            report = await ask_agent("NABIL", "http://testserver", http)
            assert "Synthetic NABIL test report." in report
            assert SOURCE_URL in report
            assert not http.is_closed
            metadata = await ask_agent_report("NABIL", "http://testserver", http)
            assert metadata["source_domains"] == ["merolagani.com", "sharesansar.com"]
            assert metadata["search_queries"] == ["NABIL news", "NABIL financial results"]

    asyncio.run(scenario())


def test_a2a_empty_query_and_failed_research_have_correct_states():
    async def scenario():
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(429, json={"error": "fixture"})

        app = create_app(NepseResearchAgent("test-key", transport=httpx.MockTransport(respond)), "http://testserver")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as http:
            for query, expected in [(" ", "TASK_STATE_INPUT_REQUIRED"), ("NABIL", "TASK_STATE_FAILED")]:
                request = SendMessageRequest(message=new_text_message(query, role=Role.ROLE_USER))
                response = await http.post("/", headers={"A2A-Version": "1.0"}, json={
                    "jsonrpc": "2.0", "id": "test", "method": "SendMessage", "params": MessageToDict(request),
                })
                task = response.json()["result"]["task"]
                assert task["status"]["state"] == expected
                assert not task.get("artifacts")
            assert len(calls) == 1
            with pytest.raises(ResearchError, match="quota"):
                await ask_agent("NABIL", "http://testserver", http)
            assert not http.is_closed

    asyncio.run(scenario())


def test_a2a_streaming_finishes_with_artifacts_and_completed_status():
    async def scenario():
        provider = httpx.MockTransport(lambda request: httpx.Response(200, json=provider_response()))
        app = create_app(NepseResearchAgent("test-key", transport=provider), "http://testserver")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as http:
            client = await create_client("http://testserver", ClientConfig(streaming=True, httpx_client=http))
            try:
                request = SendMessageRequest(message=new_text_message("NABIL", role=Role.ROLE_USER))
                events = [event async for event in client.send_message(request)]
                assert any(event.HasField("artifact_update") for event in events)
                states = [event.status_update.status.state for event in events if event.HasField("status_update")]
                assert TaskState.TASK_STATE_WORKING in states
                assert states[-1] == TaskState.TASK_STATE_COMPLETED
            finally:
                await client.close()

    asyncio.run(scenario())


def test_main_entry_point_uses_installed_sdk_and_advertises_reachable_card(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NEPSE_PROVIDER", "openai")
    module = runpy.run_path("main.py", run_name="entry_point_test")
    assert module["agent_card"].supported_interfaces[0].url == "http://127.0.0.1:8000"
    assert module["agent_card"].default_output_modes == ["text/markdown", "application/json"]
    assert [route.path for route in module["app"].routes] == ["/.well-known/agent-card.json", "/"]


# Semantic retrieval tests use deterministic embeddings and a real persistent
# Chroma store. No provider requests or embedding-model downloads are needed.
_CASH_A = "Cash generation measures whether accounting earnings produce spendable money."
_CASH_B = "Reliable operating receipts support dividends and debt payments."
_OTHER_A = "Seasonal rainfall determines reservoir levels for electricity production."
_OTHER_B = "Weather patterns change the timing of river water supply."
_CASH_TEXT = _CASH_A + " " + _CASH_B
_OTHER_TEXT = _OTHER_A + " " + _OTHER_B
_SEMANTIC_QUERY = "Do reported profits turn into real money?"


class _FixtureEmbedder:
    def __init__(self):
        self.document_calls = 0
        self.query_texts = []
        self.vectors = {
            _CASH_A: [1.0, 0.0, 0.0], _CASH_B: [1.0, 0.0, 0.0],
            _CASH_TEXT: [1.0, 0.0, 0.0], _SEMANTIC_QUERY: [1.0, 0.0, 0.0],
            _OTHER_A: [0.0, 1.0, 0.0], _OTHER_B: [0.0, 1.0, 0.0],
            _OTHER_TEXT: [0.0, 1.0, 0.0],
        }

    def embed_documents(self, texts):
        import numpy as np
        self.document_calls += 1
        return np.asarray([self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts])

    def embed_query(self, text):
        import numpy as np
        self.query_texts.append(text)
        return np.asarray([self.vectors.get(text, [0.0, 0.0, 1.0])])


def _fixture_store(tmp_path, monkeypatch, texts=None, model="fixture-model"):
    from types import SimpleNamespace
    from nepse_agent.rag_engine import BookRAGStore
    path = tmp_path / "book.pdf"
    if not path.exists():
        path.write_bytes(b"fixture PDF version 1")
    texts = texts or [_CASH_TEXT + " " + _OTHER_TEXT]
    monkeypatch.setenv("NEPSE_CHUNK_MIN_WORDS", "1")
    monkeypatch.setenv("NEPSE_RAG_MIN_SIMILARITY", "0.45")
    monkeypatch.setattr(
        "nepse_agent.rag_engine.PdfReader",
        lambda path: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in texts]),
    )
    embedder = _FixtureEmbedder()
    return BookRAGStore(str(tmp_path), embedding_model=model, embedder=embedder), embedder


def test_semantic_boundaries_follow_meaning_and_preserve_page_content(tmp_path, monkeypatch):
    store, _ = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    assert [chunk.content for chunk in store.chunks] == [_CASH_TEXT, _OTHER_TEXT]
    assert all(chunk.page_number == 1 for chunk in store.chunks)
    assert " ".join(chunk.content for chunk in store.chunks) == _CASH_TEXT + " " + _OTHER_TEXT
    # Unpunctuated tables still fit within the embedding input size.
    long_text = " ".join(["unpunctuated"] * 500)
    chunks = store._semantic_chunks(long_text)
    assert all(len(chunk.split()) <= store.max_words for chunk in chunks)
    assert " ".join(chunks) == long_text


def test_cosine_vector_search_matches_paraphrase_filters_noise_and_logs_text(tmp_path, monkeypatch, caplog):
    store, embedder = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    with caplog.at_level("INFO", logger="nepse_agent.rag_engine"):
        result = store.retrieve(_SEMANTIC_QUERY, top_k=2)
    assert len(result) == 1
    assert result[0].content == _CASH_TEXT
    assert result[0].similarity_score == pytest.approx(1.0)
    assert embedder.query_texts == [_SEMANTIC_QUERY]
    assert _SEMANTIC_QUERY in caplog.text
    assert _CASH_TEXT in caplog.text
    assert "cosine_similarity=1.0000" in caplog.text
    assert "result rejected" in caplog.text
    assert store.retrieve("   ") == []


def test_semantic_index_drops_isolated_page_numbers_and_headings(tmp_path, monkeypatch):
    store, _ = _fixture_store(tmp_path, monkeypatch, texts=[_CASH_TEXT + "\n\n9"])
    store.load_or_build()
    assert [chunk.content for chunk in store.chunks] == [_CASH_TEXT]
    assert store._semantic_chunks("9") == []
    assert store._semantic_chunks("Chapter 2") == []


def test_retrieval_backfills_after_rejecting_a_high_similarity_page_number(tmp_path, monkeypatch, caplog):
    from nepse_agent.rag_engine import DocumentChunk
    store, embedder = _fixture_store(tmp_path, monkeypatch, texts=[_CASH_TEXT])
    store.load_or_build()
    embedder.vectors["9"] = [1.0, 0.0, 0.0]
    embedder.vectors[_CASH_TEXT] = [0.9, 0.1, 0.0]
    store.chunks.insert(0, DocumentChunk("page-number", "book.pdf", "Chapter 2", 10, "9"))
    store._chunks_by_id = {chunk.chunk_id: chunk for chunk in store.chunks}
    store._index_vectors(store.chunks, rebuild=True)
    with caplog.at_level("INFO", logger="nepse_agent.rag_engine"):
        result = store.retrieve(_SEMANTIC_QUERY, top_k=1)
    assert len(result) == 1
    assert result[0].content == _CASH_TEXT
    assert "id=page-number reason=short_or_nontext" in caplog.text


def test_analysis_retrieval_prioritizes_framework_and_cleans_source_urls(tmp_path, monkeypatch):
    from nepse_agent.rag_engine import DocumentChunk
    store, _ = _fixture_store(tmp_path, monkeypatch)
    queries = []
    def retrieve(query, top_k):
        queries.append(query)
        return [DocumentChunk(str(len(queries)), "book.pdf", "Framework", 1, query, 0.8)]
    monkeypatch.setattr(store, "retrieve", retrieve)
    evidence = (
        "Researched at: 2026-10-09T12:24:31+05:45 (Asia/Kathmandu)\n\n"
        "Union Hydropower Limited has operating cash flow and valuation concerns "
        "[source](https://vertexaisearch.cloud.google.com/grounding-api-redirect/long-opaque-token)."
    )
    chunks = store.retrieve_for_analysis("unhpl", evidence=evidence, top_k=8)
    assert "unhpl" not in queries
    assert not any("https://" in query or "long-opaque-token" in query or "Researched at:" in query for query in queries)
    selected_text = " ".join(chunk.content for chunk in chunks)
    for concept in ["operating cash flow", "financial leverage", "intrinsic value", "return on equity", "due diligence"]:
        assert concept in selected_text


def test_persisted_vectors_reload_without_reembedding_books(tmp_path, monkeypatch):
    store, _ = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    reloaded, embedder = _fixture_store(tmp_path, monkeypatch)
    reloaded.load_or_build()
    assert embedder.document_calls == 0
    assert reloaded.retrieve(_SEMANTIC_QUERY)[0].content == _CASH_TEXT


def test_changed_books_remove_obsolete_results_and_changed_models_reindex(tmp_path, monkeypatch):
    store, _ = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    old_collection = store._collection.name
    (tmp_path / "book.pdf").write_bytes(b"fixture PDF version 2")
    changed, _ = _fixture_store(tmp_path, monkeypatch, texts=[_OTHER_TEXT])
    changed.load_or_build()
    assert len(changed.chunks) == 1
    assert changed._collection.name != old_collection
    assert changed.retrieve(_SEMANTIC_QUERY) == []
    new_model, embedder = _fixture_store(tmp_path, monkeypatch, texts=[_OTHER_TEXT], model="fixture-model-v2")
    new_model.load_or_build()
    assert new_model._collection.name != changed._collection.name
    assert embedder.document_calls > 0


def test_missing_vector_database_recovers_from_cached_passages(tmp_path, monkeypatch):
    from nepse_agent.rag_engine import BookRAGStore
    store, _ = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    monkeypatch.setattr(
        "nepse_agent.rag_engine.PdfReader",
        lambda path: pytest.fail("Cached passages should avoid PDF extraction"),
    )
    recovered = BookRAGStore(str(tmp_path), vector_db_path=str(tmp_path / "replacement-db"),
                             embedding_model="fixture-model", embedder=_FixtureEmbedder())
    recovered.load_or_build()
    assert recovered.retrieve(_SEMANTIC_QUERY)[0].content == _CASH_TEXT


@pytest.mark.parametrize("version", [2, 3])
def test_legacy_index_migration_preserves_ocr_and_page_references(version, tmp_path, monkeypatch):
    store, _ = _fixture_store(tmp_path, monkeypatch, texts=[_CASH_TEXT, ""])
    (tmp_path / ".rag_index.json").write_text(json.dumps({"version": version, "ocr_enabled": True}))
    calls = []
    def ocr(path, digest, pages):
        calls.append(pages)
        return {"2": _OTHER_TEXT}
    monkeypatch.setattr(store, "_ocr_pages", ocr)
    store.load_or_build()
    assert calls == [[2]]
    assert {chunk.page_number for chunk in store.chunks} == {1, 2}
    assert store.book_status[0]["ocr_pages"] == [2]
    manifest = json.loads((tmp_path / ".rag_index.json").read_text())
    assert manifest["version"] == 4
    assert manifest["ocr_enabled"] is True


def test_web_response_sections_and_exact_vector_queries_are_logged(tmp_path, monkeypatch, caplog):
    store, embedder = _fixture_store(tmp_path, monkeypatch)
    store.load_or_build()
    evidence = "The verified company report describes cash generation and dividend coverage."
    with caplog.at_level("INFO", logger="nepse_agent.rag_engine"):
        store.retrieve_for_analysis("Analyze NABIL", evidence=evidence, top_k=3)
    assert evidence in embedder.query_texts
    assert evidence in caplog.text
    assert "[RAG evidence input]" in caplog.text
    assert "[RAG vector query]" in caplog.text


@pytest.mark.parametrize("text", [
    "[BOOK:invented-chunk]",
    "[BOOK:2]",
    "[BOOK:15]",  # A PDF page number is not a citation ID.
    "[BOOK:1,2]",  # Every ID in a grouped reference must be supplied.
    "[BOOK:1, 999] [BOOK:1]",
    "[BOOK:1",
    "[BOOK:] [BOOK:1]",
    "[Other.pdf | Chapter 14 | Page 55]",
    "Book-based rationale without any citation.",
])
def test_unretrieved_or_missing_book_citations_are_rejected(text, caplog):
    from nepse_agent.agent import EvidenceError, _render_book_citations
    from nepse_agent.rag_engine import DocumentChunk
    chunk = DocumentChunk("real-id", "Book.pdf", "Chapter 9", 15, _CASH_TEXT)
    with pytest.raises(EvidenceError, match=r"Allowed book citation markers for this request: \[BOOK:1\]"):
        _render_book_citations(text, [chunk], required=True)
    if text == "[BOOK:2]":
        assert "Unknown book IDs: '2'" in caplog.text
        assert "[BOOK:1]" in caplog.text


@pytest.mark.parametrize("marker", ["[BOOK:1]", "[BOOK: 1 ]", "[BOOK:real-id]"])
def test_verified_book_markers_render_exact_metadata(marker):
    from nepse_agent.agent import _render_book_citations
    from nepse_agent.rag_engine import DocumentChunk
    chunk = DocumentChunk("real-id", "Book.pdf", "Chapter 9", 15, _CASH_TEXT)
    result = _render_book_citations(f"Cash flow reasoning {marker}", [chunk], required=True)
    assert result == "Cash flow reasoning [Book.pdf | Chapter 9 | PDF page 15]"


def test_grouped_book_markers_map_only_to_supplied_passages():
    from nepse_agent.agent import _render_book_citations
    from nepse_agent.rag_engine import DocumentChunk
    chunks = [
        DocumentChunk("Module 3_Fundamental Analysis.pdf_131_1", "Book.pdf", "Due Diligence", 131, _CASH_TEXT),
        DocumentChunk("Module 3_Fundamental Analysis.pdf_145_2", "Book.pdf", "ROE", 145, _CASH_TEXT),
    ]
    result = _render_book_citations("Reasoning [BOOK:2, 1, 2]", chunks, required=True)
    assert result == "Reasoning [Book.pdf | ROE | PDF page 145] [Book.pdf | Due Diligence | PDF page 131]"


@pytest.mark.parametrize("text", ["[BOOK:1]", "[BOOK:original-id]", "[BOOK:]"])
def test_book_markers_are_rejected_when_no_passages_were_supplied(text):
    from nepse_agent.agent import EvidenceError, _render_book_citations
    with pytest.raises(EvidenceError, match="Allowed book citations: NONE"):
        _render_book_citations(text, [], required=False)


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_web_first_analysis_retrieves_from_response_and_exposes_book_sources(provider, monkeypatch, caplog):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from nepse_agent.rag_engine import DocumentChunk
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    chunk_id = "Module 3_Fundamental Analysis.pdf_131_1"
    chunk = DocumentChunk(chunk_id, "Book.pdf", "Cash flow", 3, _CASH_TEXT, 0.85)
    retrievals = []
    def retrieve(query, evidence="", top_k=8):
        retrievals.append((query, evidence))
        return [chunk]
    monkeypatch.setattr(
        "nepse_agent.rag_engine.get_rag_store",
        lambda: SimpleNamespace(retrieve_for_analysis=retrieve),
    )
    prompts = []
    first_text = "The verified operating cash flow supports net profit and dividend payments."
    final_text = "[MODERATE / FAIR VALUE (HOLD)] Cash flow reasoning [BOOK:1]. Company evidence [WEB:1] [WEB:2]."
    def respond(request):
        payload = json.loads(request.content)
        instructions = payload["instructions"] if provider == "openai" else payload["systemInstruction"]["parts"][0]["text"]
        prompts.append(instructions)
        text = first_text if len(prompts) == 1 else final_text
        assert bool(payload.get("tools")) is (len(prompts) == 1)
        if provider == "openai":
            data = provider_response()
            part = data["output"][1]["content"][0]
            part["text"] = text
            for annotation in part["annotations"]:
                annotation.update(start_index=0, end_index=len(text))
            if len(prompts) > 1:
                data["output"] = [data["output"][1]]
                part["annotations"] = []
                assert "tool_choice" not in payload
            return httpx.Response(200, json=data)
        if len(prompts) > 1:
            return httpx.Response(200, json={"candidates": [{
                "finishReason": "STOP", "content": {"role": "model", "parts": [{"text": text}]},
            }]})
        return httpx.Response(200, json={"candidates": [{
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": text}]},
            "groundingMetadata": {
                "webSearchQueries": ["NABIL financial results"],
                "groundingChunks": [
                    {"web": {"uri": SOURCE_URL, "title": "Company source"}},
                    {"web": {"uri": "https://www.sharesansar.com/company/NABIL", "title": "ShareSansar"}},
                ],
                "groundingSupports": [{
                    "segment": {"startIndex": 0, "endIndex": len(text.encode("utf-8"))},
                    "groundingChunkIndices": [0, 1],
                }],
            },
        }]})
    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond), provider=provider)
    monkeypatch.setattr("nepse_agent.agent.time", SimpleNamespace(perf_counter=Mock(side_effect=[100.0, 112.5])))
    async def scenario():
        app = create_app(agent, "http://testserver")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            return await ask_agent_report("NABIL", "http://testserver", client)
    with caplog.at_level("INFO", logger="nepse_agent"):
        report = SimpleNamespace(**asyncio.run(scenario()))
    assert len(prompts) == 2
    assert "[BOOK:1]" not in prompts[0]
    assert "web-evidence stage" in prompts[0]
    assert first_text in prompts[1]
    assert "[BOOK:1]" in prompts[1]
    assert "Allowed book citation markers for this request: [BOOK:1]" in prompts[1]
    assert chunk_id not in prompts[1]
    assert len(retrievals) == 1
    assert first_text in retrievals[0][1]
    assert report.book_sources[0]["chunk_id"] == chunk_id
    assert report.book_sources[0]["citation_id"] == "1"
    assert report.book_sources[0]["similarity_score"] == 0.85
    assert "[Book.pdf | Cash flow | PDF page 3]" in report.markdown
    assert "[RAG model context]" in caplog.text
    assert f"[RAG citation map] [BOOK:1] -> chunk_id='{chunk_id}'" in caplog.text
    assert "[BOOK:" not in report.markdown
    assert report.elapsed_seconds == 12.5
    assert report.source_domains == ["merolagani.com", "sharesansar.com"]
    assert len(report.search_queries) == (2 if provider == "openai" else 1)
    assert "[WEB:" not in report.markdown


def test_default_web_first_keeps_market_summaries_as_one_search(monkeypatch):
    monkeypatch.delenv("NEPSE_RAG_WEB_FIRST", raising=False)
    assert NepseResearchAgent._uses_web_first("NABIL") is True
    assert NepseResearchAgent._uses_web_first("Analyze UNL cash flow") is True
    assert NepseResearchAgent._uses_web_first("Give me today's NEPSE market summary") is False
    assert NepseResearchAgent._uses_web_first("Latest Nepal economic news") is False


@pytest.mark.parametrize("recover", [False, True])
def test_invented_book_citations_retry_and_never_reach_the_user(recover, monkeypatch):
    from types import SimpleNamespace
    from nepse_agent.agent import EvidenceError
    from nepse_agent.rag_engine import DocumentChunk
    monkeypatch.setenv("NEPSE_RAG_WEB_FIRST", "1")
    chunk = DocumentChunk("real-id", "Book.pdf", "Cash flow", 3, _CASH_TEXT, 0.85)
    monkeypatch.setattr(
        "nepse_agent.rag_engine.get_rag_store",
        lambda: SimpleNamespace(retrieve_for_analysis=lambda *args, **kwargs: [chunk]),
    )
    prompts = []
    def respond(request):
        payload = json.loads(request.content)
        prompts.append(payload["instructions"])
        if len(prompts) == 1:
            text = "Verified company financial evidence."
        else:
            book_id = "1" if recover and len(prompts) == 3 else "invented"
            text = f"Financial rationale [BOOK:{book_id}] [WEB:1] [WEB:2]"
        data = provider_response()
        part = data["output"][1]["content"][0]
        part["text"] = text
        for annotation in part["annotations"]:
            annotation.update(start_index=0, end_index=len(text))
        if len(prompts) > 1:
            data["output"] = [data["output"][1]]
            part["annotations"] = []
        return httpx.Response(200, json=data)
    agent = NepseResearchAgent("fixture-key", "fixture-model", httpx.MockTransport(respond))
    if recover:
        report = asyncio.run(agent.research("NABIL"))
        assert "[Book.pdf | Cash flow | PDF page 3]" in report.markdown
        assert "invented" not in report.markdown
    else:
        with pytest.raises(EvidenceError, match="book passages that were not retrieved"):
            asyncio.run(agent.research("NABIL"))
    assert len(prompts) == 3
    assert "previous response failed evidence validation" in prompts[-1]
    assert "Invalid book IDs: 'invented'" in prompts[-1]
    assert "Allowed book citation markers for this request: [BOOK:1]" in prompts[-1]
