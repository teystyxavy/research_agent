from datetime import datetime
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import LLM_MODEL, LLM_TEMPERATURE, MAX_ITERATIONS
from .state import AgentState
from .tools import TOOLS
from .prompts import RESEARCH_AGENT_PROMPT

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
llm_with_tools = llm.bind_tools(TOOLS)
_tool_node = ToolNode(TOOLS)

# nodes
def research_agent_node(state: AgentState) -> dict:
    """brain node, reasons what to do next"""
    system_prompt = RESEARCH_AGENT_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"), topic=state["topic"])
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iteration_count": state.get("iteration_count", 0) + 1}

def tool_executor_node(state: AgentState) -> dict:
    return _tool_node.invoke(state)

def human_review_node(_: AgentState) -> dict:
    # Hard stop after MAX_ITERATIONS — no real HITL implemented
    return {"approved": False}

# edges
def should_continue(state: AgentState) -> Literal["tools", "human_review", "end"]:
    """check if the agent should continue or stop"""
    last_message = state["messages"][-1]
    iterations = state.get("iteration_count", 0)

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if iterations >= MAX_ITERATIONS:
            return "human_review"
        return "tools"
    return "end"

def after_human_review(state: AgentState) -> Literal["tools", "end"]:
    """Route based on human approval."""
    if state.get("approved", False):
        return "tools"
    return "end"

# build graph
def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", research_agent_node)
    graph.add_node("tools", tool_executor_node)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "human_review": "human_review", "end": END}
    )

    graph.add_edge("tools", "agent")

    graph.add_conditional_edges(
        "human_review",
        after_human_review,
        {"tools": "tools", "end": END}
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent_graph = build_agent_graph()
