import streamlit as st
import asyncio
from dataclasses import asdict
from streamlit.components.v1 import html
from nepse_agent.client import AgentUnavailableError, ask_agent_report
from nepse_agent.agent import NepseResearchAgent, ResearchError

st.set_page_config(page_title="NEPSE Agent Chat", page_icon="📈", layout="centered")
st.title("📈 NEPSE Market & Web Research Assistant")
st.caption("Ask for today's market summary, company research, comparisons, news, or another research question.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("search_suggestions_html"):
            html(msg["search_suggestions_html"], height=120, scrolling=True)

# User prompt
prompt = st.chat_input("Ask a question, e.g. Give me today's market summary")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching your question across web sources..."):
            try:
                # Try calling via A2A Server first
                try:
                    report_data = asyncio.run(ask_agent_report(prompt, "http://127.0.0.1:10000"))
                except AgentUnavailableError:
                    st.info("The A2A server is unavailable; running research directly.")
                    report_data = asdict(asyncio.run(NepseResearchAgent().research(prompt)))
            except (ResearchError, ValueError) as exc:
                st.error(str(exc))
            else:
                response = report_data["markdown"]
                suggestions = report_data.get("search_suggestions_html", "")
                st.markdown(response)
                if suggestions:
                    html(suggestions, height=120, scrolling=True)
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "search_suggestions_html": suggestions,
                })
