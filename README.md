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

## NEPSE Research Agent

Research a NEPSE symbol for recent news, the latest available market information,
company fundamentals and corporate actions, with dates and clickable sources.
Run these commands from the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key'
python -m nepse_agent research "NABIL"
```

To expose the researcher to other A2A agents, run `python -m nepse_agent serve`.
Then query it from another terminal with `python -m nepse_agent ask "NABIL"`.
See [the setup and architecture guide](nepse_agent/README.md) for configuration,
report contents, data limitations and extension points.
