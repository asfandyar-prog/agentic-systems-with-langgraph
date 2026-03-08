from langgraph.graph import StateGraph,MessagesState,START,END


def mock_llm(state:MessagesState):
    return {"message":[{"role":"ai","content":"hello world"}]}


graph= StateGraph(MessagesState)

graph.add_node(mock_llm)
graph.add_edge(START,"mock_llm")
graph.add_edge("mock_llm",END)

graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
