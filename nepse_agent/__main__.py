"""Run direct research, start an A2A server, or query an A2A server."""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

import httpx

from .agent import NepseResearchAgent, ResearchError


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer market and research questions with sourced web information.")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser("research", help="Research a question or stock symbol directly")
    research.add_argument("query", help='For example: "Give me today\'s market summary", "Compare NABIL and EBL", or "NABIL"')
    research.add_argument("--json", action="store_true", help="Return report text and source metadata as JSON")
    serve = commands.add_parser("serve", help="Start the A2A agent server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=10000)
    serve.add_argument("--public-url", help="Address that A2A clients use to reach this server")
    ask = commands.add_parser("ask", help="Query an already running A2A server")
    ask.add_argument("query")
    ask.add_argument("--url", default="http://127.0.0.1:10000")
    args = parser.parse_args()
    try:
        if args.command == "research":
            report = asyncio.run(NepseResearchAgent().research(args.query))
            print(json.dumps(asdict(report), ensure_ascii=False, indent=2) if args.json else report.markdown)
        elif args.command == "serve":
            import uvicorn
            from .server import create_app

            public_url = args.public_url or f"http://{args.host}:{args.port}"
            uvicorn.run(create_app(public_url=public_url), host=args.host, port=args.port)
        else:
            from .client import ask_agent

            print(asyncio.run(ask_agent(args.query, args.url)))
    except (ResearchError, ValueError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
