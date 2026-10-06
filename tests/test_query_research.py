"""Non-symbol questions through the real provider adapters and A2A transport.

Provider replies are synthetic; these tests do not verify live market figures.
"""

import asyncio
import json

import httpx
import pytest

from nepse_agent.agent import NepseResearchAgent
from nepse_agent.client import ask_agent_report
from nepse_agent.server import create_app


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


def request_content(provider, request):
    payload = json.loads(request.content)
    if provider == "gemini":
        assert payload["tools"] == [{"googleSearch": {}}]
        return payload["contents"][0]["parts"][0]["text"], payload["systemInstruction"]["parts"][0]["text"]
    assert payload["tool_choice"] == "required"
    assert payload["tools"] == [{"type": "web_search", "external_web_access": True}]
    return payload["input"], payload["instructions"]


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
