# A2A Protocol Implementation & Samples

This repository contains exploration and implementations of the **Agent-to-Agent (A2A) Protocol**, enabling autonomous AI agents to discover, communicate, and collaborate across standard protocols (JSON-RPC, REST, SSE, and gRPC).

## Features
- **Agent Discovery**: Uses standardized `AgentCard` manifests for capabilities and skills.
- **Inter-Agent Communication**: Streaming and non-streaming task execution.
- **Multi-Agent Orchestration**: Host agents delegating work to specialized remote agents.

## Getting Started

1. Activate your virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the HelloWorld Agent sample:
   ```bash
   cd a2a-samples/samples/python/agents/helloworld
   python __main__.py
   ```
