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


# example with different State schemas

from typing import TypedDict
from langgraph.graph.state import StateGraph,START,END

class SubGraphState(TypedDict):
    #Note that none of these fielsd keys are shared with the parent graph state
    bar:str
    baz:str 

def sub_graph_node_1(state:SubGraphState):
    return {"bar":"hello world" +state["bar"],"baz":"baz value"}

def sub_graph_node_2(state:SubGraphState):
    return {"bar":state["bar"] + state["baz"]}

subgraph_builder=StateGraph(SubGraphState)
subgraph_builder.add_node("node-1",sub_graph_node_1)
subgraph_builder.add_node("node-2",sub_graph_node_2 )
subgraph_builder.add_edge(START,"node-1")
subgraph_builder.add_edge("node-1","node-2")    
subgraph_builder.add_edge("node-2",END)
subgraph=subgraph_builder.compile() 
