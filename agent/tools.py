import os
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
from .rag import RAGAgent

from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TAVILY_MAX_RESULTS, SEARCH_RESULT_MAX_CHARS, REPORTS_DIR, RAG_K, CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_BASE_NAME

search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
rag = RAGAgent(model=EMBEDDING_MODEL, name=KNOWLEDGE_BASE_NAME, persist_dir=CHROMA_PERSIST_DIR,
               chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

@tool
def web_search(query: str) -> str:
    """Search the web for information on a given query."""
    results = search_tool.invoke(query)
    formatted = []
    for r in results:
        content = r['content'][:SEARCH_RESULT_MAX_CHARS]
        formatted.append(f"Source: {r['url']}\n{content}\n")
    return "\n---\n".join(formatted)

@tool
def save_report(content: str, filename: str = None) -> str:
    """Save the final research report to disk.
    Call this only when the research is complete and the report is fully written."""
    if not filename:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(f"{REPORTS_DIR}/{filename}", "w") as f:
        f.write(content)
    return f"Report saved to {filename}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Use for any numerical calculations during research."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    
@tool
def retrieve_from_knowledge_base(query: str) ->  str:
    """ Searches private knowledge base for relevant documents"""
    results = rag.retrieve(query, k=RAG_K)
    if not results:
        return "No relevant information found in knowledge base."
    formatted = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source} (score: {score:.2f})\n{doc.page_content}")
    return "\n---\n".join(formatted)

# Register all tools the agent can call
TOOLS = [web_search, save_report, calculate, retrieve_from_knowledge_base]