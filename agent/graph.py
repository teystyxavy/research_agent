import os
from datetime import datetime
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .tools import TOOLS
from .prompts import RESEARCH_AGENT_PROMPT

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

# nodes

def research_agent_node(state: AgentState) -> dict:
    """brain node, reasons what tdo do next"""
    system_prompt = RESEARCH_AGENT_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"), topic=state["topic"])
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iteration_count": state.get("iteration_count", 0) + 1}

def tool_executor_node(state: AgentState) -> dict:
    tool_node = ToolNode(TOOLS)
    return tool_node.invoke(state)

def human_review_node(state: AgentState) -> dict:
    iterations = state.get("iteration_count", 0)
    return {"approved": iterations >= 3}

# edges 
def should_continue(state: AgentState) -> Literal["tools", "human_review", "end"]:
    """check if the agent should continue or stop"""
    last_message = state["messages"][-1]
    iterations = state.get("iteration_count", 0)

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if iterations >= 6:
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

    # add nodes
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
