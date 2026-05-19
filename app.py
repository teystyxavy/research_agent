import os
import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import agent_graph
from agent.config import REPORTS_DIR, THREAD_ID
from agent.tools import rag

st.set_page_config(page_title="Research Agent", page_icon="🔍", layout="wide")
st.title("🔍 Agentic Research Assistant")
st.caption("Powered by LangGraph · Gemini 2.0 Flash · Multi-tool autonomous agent")

# Sidebar — config
with st.sidebar:
    st.header("Agent Config")
    st.caption("Configure via `.env` — see `agent/config.py`")
    st.divider()
    st.markdown("**Tools available:**")
    st.markdown("- 🔍 Web search (Tavily)")
    st.markdown("- 💾 Save report")
    st.markdown("- 🧮 Calculator")
    st.divider()
    files = st.file_uploader("Upload documents to knowledge base", type=["txt", "pdf", "md"], accept_multiple_files=True)
    if files:
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = set()
        new_files = [f for f in files if f.name not in st.session_state.ingested_files]
        if new_files:
            with st.spinner(f"Ingesting {len(new_files)} file(s)..."):
                for f in new_files:
                    rag.add_document(f.read(), source=f.name)
                    st.session_state.ingested_files.add(f.name)
            st.success(f"Added {len(new_files)} file(s) to knowledge base.")
    if st.button("Clear session"):
        st.session_state.clear()
        st.rerun()

# Main interface
topic = st.text_input(
    "Research topic",
    placeholder="e.g. 'The economic impact of generative AI on software jobs in 2025'"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("Run Agent ▶", type="primary", use_container_width=True)

if run_button and topic:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    thread_config = {"configurable": {"thread_id": THREAD_ID}}

    # Stream the agent's execution step by step
    with st.spinner("Agent is researching..."):
        step_container = st.container()
        events = agent_graph.stream(
            {
                "messages": [HumanMessage(content=f"Research this topic: {topic}")],
                "topic": topic,
                "research_notes": [],
                "final_report": "",
                "iteration_count": 0,
                "approved": False,
            },
            config=thread_config,
            stream_mode="values"
        )

        for i, event in enumerate(events):
            messages = event.get("messages", [])
            if messages:
                last = messages[-1]
                node_name = "Agent" if hasattr(last, "tool_calls") else "Tool"

                with step_container.expander(
                    f"Step {i+1} — {node_name}", expanded=(i < 3)
                ):
                    st.markdown(f"**Iterations:** {event.get('iteration_count', 0)}")
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        for tc in last.tool_calls:
                            st.code(f"Tool: {tc['name']}\nInput: {tc['args']}", language="json")
                    elif hasattr(last, "content"):
                        st.markdown(last.content[:1000])

    st.success("Research complete!")

    # Show final state
    final_state = agent_graph.get_state(thread_config)
    if final_state and final_state.values.get("messages"):
        last_msg = final_state.values["messages"][-1]
        st.subheader("📄 Final Report")
        st.markdown(last_msg.content)

elif run_button and not topic:
    st.warning("Please enter a research topic.")