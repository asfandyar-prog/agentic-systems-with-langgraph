from typing import TypedDict
from langgraph.graph.state import StateGraph, START, END


class SubgraphState(TypedDict):
    bar:str
    #SubgraphState can have any fields, but for this example we just have one field called "bar"

def mock_subgraph(state:SubgraphState):
    return {"bar":"hello world" +state["bar"]}


subgraph_builder=StateGraph(SubgraphState)
subgraph_builder.add_node(mock_subgraph)
subgraph_builder.add_edge(START,"mock_subgraph")
subgraph=subgraph_builder.compile()



#Parent Graph

class State(TypedDict):
    foo:str
    
def call_subgraph(state:State):
    #Transform the state to the subgraph state
    subgraph_output=subgraph.invoke({"bar":state["foo"]})
    #Transform the subgraph output back to the parent graph state
    return {"foo":subgraph_output["bar"]}

graph_builder= StateGraph(State)
graph_builder.add_node("node-1",call_subgraph)
graph_builder.add_edge(START,"node-1")
graph=graph_builder.compile()