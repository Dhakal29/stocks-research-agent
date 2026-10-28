import re
import streamlit as st
import asyncio
from dataclasses import asdict
from nepse_agent.client import AgentUnavailableError, ask_agent_report
from nepse_agent.agent import NepseResearchAgent, ResearchError, configure_research_logging

configure_research_logging()
st.set_page_config(page_title="NEPSE Research Agent", page_icon="📈", layout="centered")
st.title("📈 NEPSE Research Agent")
st.caption("Ask for today's market summary, company research, comparisons, news, or another research question.")

# Initialize chat history and benchmark state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None
if "benchmark_company" not in st.session_state:
    st.session_state.benchmark_company = ""


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


def _extract_company_from_messages() -> str:
    """Try to infer the most recently researched company/ticker from chat history."""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            text = msg["content"].strip()
            # Prefer bare tickers (all-caps, 2-12 chars) or the first capitalised word
            ticker_match = re.search(r"\b([A-Z]{2,12})\b", text)
            if ticker_match:
                return ticker_match.group(1)
            return text[:60]  # fallback: truncated query
    return ""


# ── Sidebar: Benchmarking ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Financial Benchmarking")
    st.caption(
        "Compare revenue growth, gross margin, R&D spend, and cash conversion "
        "cycle for a company against its sector peers using the latest financial "
        "statements from the web."
    )

    default_company = _extract_company_from_messages()
    benchmark_input = st.text_input(
        "Company name or ticker",
        value=default_company,
        placeholder="e.g. NABIL, AAPL, Reliance",
        key="benchmark_input",
    )

    run_benchmark = st.button("📊 Run Benchmarking", use_container_width=True)

    if run_benchmark:
        company = benchmark_input.strip()
        if not company:
            st.error("Enter a company name or ticker before running benchmarking.")
        else:
            with st.spinner(f"Searching web for latest financials of **{company}**…"):
                import time
                from nepse_agent.benchmarking import run_benchmark as _run_benchmark

                try:
                    bm = _run_benchmark(company)
                    st.session_state.benchmark_result = bm
                    st.session_state.benchmark_company = company
                except Exception as exc:
                    st.error(f"Benchmarking failed: {exc}")
                    st.session_state.benchmark_result = None

    # Render cached benchmark result (survives reruns)
    if st.session_state.benchmark_result is not None:
        bm = st.session_state.benchmark_result
        st.success(f"✅ Benchmark ready for **{st.session_state.benchmark_company}**")
        st.caption(f"⏱️ Retrieved in {bm.elapsed_seconds:.2f}s · {bm.researched_at}")

        with st.expander("📋 View Benchmarking Table", expanded=True):
            render_clean_report(bm.markdown)

        if st.button("🗑️ Clear benchmark", use_container_width=True):
            st.session_state.benchmark_result = None
            st.session_state.benchmark_company = ""
            st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────

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
