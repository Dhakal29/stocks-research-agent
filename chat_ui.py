import streamlit as st
import asyncio
from dataclasses import asdict
from nepse_agent.client import AgentUnavailableError, ask_agent_report
from nepse_agent.agent import NepseResearchAgent, ResearchError, configure_research_logging

configure_research_logging()
st.set_page_config(page_title="NEPSE Research Agent", page_icon="📈", layout="centered")
st.title("📈 NEPSE Research Agent")
st.caption("Ask for today's market summary, company research, comparisons, news, or another research question.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_clean_report(markdown_text: str, elapsed_seconds: float | None = None) -> None:
    # Display latency badge if available
    if elapsed_seconds is not None and elapsed_seconds > 0:
        st.caption(f"⏱️ **Response latency:** {elapsed_seconds:.2f}s")

    # Separate the main analysis from raw cited source lists to keep the UI clean
    if "## Cited sources" in markdown_text:
        main_content, sources_content = markdown_text.split("## Cited sources", 1)
        st.markdown(main_content.strip())
        with st.expander("🔗 View Web Sources & Citations", expanded=False):
            st.markdown(sources_content.strip())
    else:
        st.markdown(markdown_text)

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_clean_report(msg["content"], msg.get("elapsed_seconds"))

# User prompt
prompt = st.chat_input("Ask a question, e.g. Analyze NABIL based on fundamental books")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing stock fundamentals against training books..."):
            import time
            start_ts = time.perf_counter()
            try:
                # Try calling via A2A Server first
                try:
                    report_data = asyncio.run(ask_agent_report(prompt, "http://127.0.0.1:10000"))
                except AgentUnavailableError:
                    report_data = asdict(asyncio.run(NepseResearchAgent().research(prompt)))
            except (ResearchError, ValueError) as exc:
                st.error(str(exc))
            else:
                elapsed = report_data.get("elapsed_seconds") or round(time.perf_counter() - start_ts, 2)
                response = report_data["markdown"]
                render_clean_report(response, elapsed)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "elapsed_seconds": elapsed,
                })
