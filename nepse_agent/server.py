"""Expose the research agent using the installed A2A 1.x SDK."""

import logging
from dataclasses import asdict

from a2a.helpers import new_data_part, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from starlette.applications import Starlette

from .agent import NepseResearchAgent, ResearchError, validate_query


class NepseAgentExecutor(AgentExecutor):
    def __init__(self, agent: NepseResearchAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ValueError("A text message is required.")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        try:
            query = validate_query(context.get_user_input())
        except ValueError as exc:
            await updater.update_status(TaskState.TASK_STATE_INPUT_REQUIRED, new_text_message(str(exc)))
            return
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Searching company data, financial reports and recent news..."),
        )
        try:
            report = await self.agent.research(query)
        except ResearchError as exc:
            await updater.update_status(TaskState.TASK_STATE_FAILED, new_text_message(str(exc)))
            return
        except Exception:
            logging.getLogger(__name__).exception("NEPSE research failed")
            await updater.update_status(
                TaskState.TASK_STATE_FAILED, new_text_message("Research failed. Check the server logs.")
            )
            return
        await updater.add_artifact(
            parts=[new_text_part(report.markdown, media_type="text/markdown")],
            name="nepse-research-report",
        )
        await updater.add_artifact(
            parts=[new_data_part(asdict(report), media_type="application/json")],
            name="nepse-research-data",
        )
        await updater.update_status(TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not supported by this research agent.")


def create_app(
    agent: NepseResearchAgent | None = None,
    public_url: str = "http://127.0.0.1:10000",
) -> Starlette:
    card = AgentCard(
        name="NEPSE Research Agent",
        description="Research NEPSE symbols using current web evidence, news and company fundamentals.",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/markdown", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=public_url, protocol_version="1.0")
        ],
        skills=[AgentSkill(
            id="nepse_company_research",
            name="NEPSE company research",
            description="Find sourced market information, recent news, financial metrics and corporate actions.",
            tags=["nepse", "stocks", "news", "fundamentals", "web-search"],
            examples=["NABIL", "Research EBL news and fundamentals", "What changed recently for NLIC?"],
            input_modes=["text/plain"],
            output_modes=["text/markdown", "application/json"],
        )],
    )
    handler = DefaultRequestHandler(
        agent_executor=NepseAgentExecutor(agent or NepseResearchAgent()),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))
