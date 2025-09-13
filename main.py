from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = StateGraph(MessagesState)
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

def event_loop(state: MessagesState) -> str:
    print("=== Event loop ===", state)
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop, {END: END, "execute_tools": "execute_tools"})
builder.set_entry_point("draft")

graph = builder.compile()

# print(graph.get_graph().draw_mermaid_png(output_file_path="graph.png"))

# res= graph.invoke("Write about new nano banana ai from Google. What new concepts it has used and list the affected companies")

res = graph.invoke({"messages": [HumanMessage(content="Explain Attention Is All You Need")]})

# print(res[-1].tool_calls[0]["args"]["answer"])
print(res)