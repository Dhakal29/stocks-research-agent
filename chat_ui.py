import streamlit as st
import asyncio
from nepse_agent.client import ask_agent
from nepse_agent.agent import NepseResearchAgent, ResearchError

st.set_page_config(page_title="NEPSE Agent Chat", page_icon="📈", layout="centered")
st.title("📈 NEPSE A2A Research Assistant")
st.caption("Ask questions about any NEPSE listed stock (e.g., NABIL, SHIVM, CHCL, EBL)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
prompt = st.chat_input("Enter NEPSE symbol or question:")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching company data & news..."):
            try:
                # Try calling via A2A Server first
                response = asyncio.run(ask_agent(prompt, "http://127.0.0.1:10000"))
            except Exception:
                # If A2A server is not running, run directly with agent
                st.info("ℹ️ Note: A2A server on port 10000 is offline; answering via direct agent mode.")
                report = asyncio.run(NepseResearchAgent().research(prompt))
                response = report.markdown

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    