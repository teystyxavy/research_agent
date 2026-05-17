from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    research_notes: List[str]
    final_report: str
    iteration_count: int
    approved: bool # for human in the loop
    