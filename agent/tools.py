from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime

search_tool = TavilySearchResults(max_results=5)

@tool
def web_search(query: str) -> str:
    results = search_tool.invoke(query)
    formatted = []
    for r in results:
        formatted.append(f"Source: {r['url']}\n{r['content']}\n")
    return "\n---\n".join(formatted)

@tool
def save_report(content: str, filename: str = None) -> str:
    """Save the final research report to disk.
    Call this only when the research is complete and the report is fully written."""
    if not filename:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(f"./reports/{filename}", "w") as f:
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

# Register all tools the agent can call
TOOLS = [web_search, save_report, calculate]