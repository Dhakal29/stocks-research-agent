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
from nepse_agent.client import ask_agent
from nepse_agent.server import create_app


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
                "action": {"type": "search", "sources": [
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
    assert len(report.cited_sources) == 1
    assert len(report.searched_urls) == 2
    assert report.researched_at.endswith("+05:45")


def test_citation_spans_retain_claims_and_group_multiple_sources():
    data = provider_response()
    part = data["output"][1]["content"][0]
    part["text"] = "A sourced claim must survive rendering."
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
        NepseResearchAgent()

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
    module = runpy.run_path("main.py", run_name="entry_point_test")
    assert module["agent_card"].supported_interfaces[0].url == "http://127.0.0.1:8000"
    assert module["agent_card"].default_output_modes == ["text/markdown", "application/json"]
    assert [route.path for route in module["app"].routes] == ["/.well-known/agent-card.json", "/"]
