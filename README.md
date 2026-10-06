# Agent-to-Agent (A2A) Protocol: NEPSE Research Agent

An implementation of the **Agent-to-Agent (A2A) Protocol (v1.0)** answering natural-language research questions, specializing in Nepal Stock Exchange (NEPSE) markets, companies and the economy.

The repository includes:
- **A2A Server**: Standard JSON-RPC (`SendMessage`) service exposing an `AgentCard` at `/.well-known/agent-card.json`.
- **NEPSE Research Engine**: Internet search across multiple websites using **Google Gemini** (`gemini-3.5-flash-lite`) or **OpenAI**, with dated market information, fundamentals, news and citations.
- **Interactive Chat UI**: Streamlit web chat with conversation history and automated server fallback.
- **CLI Tools & Client**: Query agents directly, via CLI, or over the network.

---

## Architecture

```
┌─────────────────────────────────┐
│     Streamlit Web Chat UI       │  (chat_ui.py)
└──────────────┬──────────────────┘
               │ (A2A JSON-RPC / SendMessage)
               ▼
┌─────────────────────────────────┐
│      NEPSE Research Agent       │  (nepse_agent/server.py on port 10000)
│   Discovery: /.well-known/      │
└──────────────┬──────────────────┘
               │
        ┌──────┴───────────────────────┐
        ▼                              ▼
┌──────────────┐              ┌────────────────┐
│ Internet     │              │ Gemini / LLM   │
│ Search Tool  │              │ Synthesis      │
└──────────────┘              └────────────────┘
```

---

## Features

- **Standard A2A Protocol**: Fully compliant `AgentCard` metadata, task state lifecycle (`TASK_STATE_WORKING` -> `TASK_STATE_COMPLETED`), and structured report artifacts.
- **Fundamentals and News**: Searches for the latest available prices, financial metrics, dated company results and corporate announcements.
- **Answers Based on the Query**: Supports today's market summary, news, sector analysis, comparisons, economic questions and other research topics. A stock symbol is optional; a bare symbol still requests a full company report.
- **Multiple Website Sources**: Requires citations from at least two publisher sites, with one further search attempt when coverage is insufficient. Configure the threshold with `NEPSE_MIN_SITES`.
- **AI-Powered Financial Insights**: Synthesizes market observations and highlights sector risks.
- **Multiple Interfaces**:
  - Interactive Web Chat UI (Streamlit)
  - HTTP JSON-RPC Server
  - CLI Direct Research (`python -m nepse_agent research`)
  - Inter-Agent Python Client (`nepse_agent.client.ask_agent`)

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd A2A_Protocol
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY="your-gemini-api-key"
   NEPSE_PROVIDER="gemini"
   NEPSE_MIN_SITES=2
   # Optional: OPENAI_API_KEY="your-openai-api-key"
   ```

---

## Running the Project

### Option 1: Interactive Chat UI (Streamlit)

Launch the web chat interface:
```bash
streamlit run chat_ui.py
```
Open `http://localhost:8501` in your browser. Ask a question such as:

- "Give me today's market summary."
- "Which NEPSE sectors performed best this week?"
- "Compare NABIL and EBL's latest financial results."
- "What are the latest IPO announcements in Nepal?"
- "How do interest rates affect Nepal's stock market?"

You can also enter a stock symbol (e.g., `NABIL`) for a full company report.
Unspecified market questions default to NEPSE; name another market explicitly
to research it. The answer follows the question's topic, timeframe, language
and requested level of detail.

---

### Option 2: Running as an A2A Server (For Multi-Agent Systems)

Start the A2A server daemon:
```bash
python -m nepse_agent serve --port 10000
```

1. **Verify Discovery (`AgentCard`)**:
   ```bash
   curl http://127.0.0.1:10000/.well-known/agent-card.json
   ```

2. **Query from another terminal using the A2A Client**:
   ```bash
   python -m nepse_agent ask "NABIL" --url http://127.0.0.1:10000
   ```

3. **Or call via direct JSON-RPC**:
   ```bash
   curl -X POST http://127.0.0.1:10000/ \
     -H "Content-Type: application/json" \
     -H "A2A-Version: 1.0" \
     -d '{
       "jsonrpc": "2.0",
       "id": "1",
       "method": "SendMessage",
       "params": {
         "message": {
           "role": "ROLE_USER",
           "parts": [{"text": "SHIVM"}]
         }
       }
     }'
   ```

---

### Option 3: Quick Direct CLI Research

Run research directly without starting an HTTP server:
```bash
python -m nepse_agent research "NABIL"
python -m nepse_agent research "Give me today's market summary"
python -m nepse_agent research "Latest Nepal economic news in Nepali"
```
Or output raw JSON data:
```bash
python -m nepse_agent research "NABIL" --json
```

The Gemini backend registers `types.Tool(google_search=types.GoogleSearch())`.
Its returned grounding metadata supplies the actual search queries and source
citations. The JSON report includes `source_domains`, `search_queries`,
`cited_sources` and Google Search suggestions for graphical clients.
The model plans searches from the user's question, starts with broad web searches,
then checks relevant primary sources and other publishers. It is not restricted
to MeroLagani or a fixed list of sites. Search retrieves relevant accessible
evidence; it cannot read the entire internet. For today's summaries, the model
is instructed to verify the actual session date, label intraday data and identify
older data when today's figures are unavailable.
See [the search-tool and source-coverage guide](nepse_agent/README.md#internet-search-tool-and-source-coverage)
for configuration and [Google's grounding documentation](https://ai.google.dev/gemini-api/docs/generate-content/google-search)
for the API contract. Prices retain their source dates; search does not guarantee
a live quote or access to paywalled financial data.

---

## Running Automated Tests

Run the test suite with pytest:
```bash
python -m pytest -q -p no:cacheprovider tests
```

---

## Project Structure

```text
.
├── .env                       # API keys (git-ignored)
├── requirements.txt           # Production dependencies
├── chat_ui.py                 # Streamlit chat interface
│
├── nepse_agent/               # NEPSE A2A Agent package
│   ├── agent.py               # Internet search tools, grounding, source coverage & synthesis
│   ├── server.py              # A2A AgentCard & JSON-RPC server routes
│   ├── client.py              # A2A client helper for inter-agent communication
│   └── __main__.py            # CLI entry point (serve, research, ask)
│
└── tests/
    └── test_nepse_agent.py    # Unit tests for A2A compliance & agent logic
```
