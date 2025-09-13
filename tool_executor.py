from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool: TavilySearch = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

def safe_execute_tools(state):
    messages = state["messages"]
    if not any(isinstance(m, AIMessage) for m in messages):
        print("⚠️ No AIMessage found, skipping tool execution")
        return state
    return execute_tools.invoke(state)

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

