"""Send a query to the research agent from another A2A application."""

import httpx
from google.protobuf.json_format import MessageToDict
from a2a.client import A2AClientError, ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from .agent import ResearchError, validate_query


class AgentUnavailableError(ResearchError):
    """The A2A endpoint could not be reached or used."""


async def ask_agent(
    query: str,
    url: str = "http://127.0.0.1:10000",
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Return a report, leaving an explicitly supplied HTTP client open."""
    return (await ask_agent_report(query, url, http_client))["markdown"]


async def ask_agent_report(
    query: str,
    url: str = "http://127.0.0.1:10000",
    http_client: httpx.AsyncClient | None = None,
) -> dict:
    """Return report metadata, including Google Search suggestions for UI rendering."""
    query = validate_query(query)
    if http_client is None:
        async with httpx.AsyncClient(timeout=480) as client:
            return await ask_agent_report(query, url, client)
    try:
        client = await create_client(
            agent=url,
            client_config=ClientConfig(
                streaming=False, httpx_client=http_client, supported_protocol_bindings=["JSONRPC"]
            ),
        )
        request = SendMessageRequest(message=new_text_message(query, role=Role.ROLE_USER))
        async for event in client.send_message(request):
            if event.HasField("task"):
                task = event.task
                if task.status.state != TaskState.TASK_STATE_COMPLETED:
                    raise ResearchError(get_message_text(task.status.message) or "The research task did not complete.")
                for artifact in task.artifacts:
                    if artifact.name == "nepse-research-data":
                        for part in artifact.parts:
                            if part.HasField("data"):
                                data = MessageToDict(part.data)
                                if isinstance(data.get("markdown"), str):
                                    return data
                for artifact in task.artifacts:
                    if artifact.name == "nepse-research-report":
                        return {"markdown": "\n".join(part.text for part in artifact.parts if part.HasField("text"))}
            elif event.HasField("message"):
                return {"markdown": get_message_text(event.message)}
    except A2AClientError as exc:
        raise AgentUnavailableError(f"Could not call the A2A agent: {exc}") from exc
    # The JSON-RPC transport shares http_client; its owner closes that client.
    raise ResearchError("The agent returned no research report.")
