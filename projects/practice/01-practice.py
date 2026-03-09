from langgraph.graph import StateGraph,MessagesState,START,END


def mock_llm(state:MessagesState):
    return {"message":[{"role":"ai","content":"hello world"}]}


graph= StateGraph(MessagesState)

graph.add_node(mock_llm)
graph.add_edge(START,"mock_llm")
graph.add_edge("mock_llm",END)

graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})



from typing import TypedDict, List, Optional

# Define the structure of a single shopping item
class Shoping(TypedDict):
    item: str
    quantity: int
    bought: Optional[bool]  # Can be True, False, or None

# Define the structure of the shopping list
class ShopingList(TypedDict):
    shopings: List[Shoping]

# Initialize a shopping list
shoping_list: ShopingList = {
    "shopings": [
        {"item": "apple", "quantity": 3, "bought": None},
        {"item": "banana", "quantity": 2, "bought": None},
        {"item": "orange", "quantity": 1, "bought": None},
    ]
}

# Function to print the shopping list
def print_shoping_list(shoping_list: ShopingList):
    for shoping in shoping_list["shopings"]:
        status = "✅" if shoping["bought"] else "❌" if shoping["bought"] == False else "⏳"
        print(f"{status} {shoping['quantity']} x {shoping['item']}")

# Function to update the shopping list
def update_shoping_list(shoping_list: ShopingList, item: str, bought: bool):
    for shoping in shoping_list["shopings"]:
        if shoping["item"] == item:
            shoping["bought"] = bought
            break

# Add a new shopping item to the list
def add_shoping_item(shoping_list: ShopingList, item: str, quantity: int):
    shoping_list["shopings"].append({"item": item, "quantity": quantity, "bought": None})

# Example Usage
print("Initial Shopping List:")
print_shoping_list(shoping_list)

print("\nUpdating 'apple' to bought...")
update_shoping_list(shoping_list, "apple", True)
print_shoping_list(shoping_list)

print("\nAdding 'grapes' to the shopping list...")
add_shoping_item(shoping_list, "grapes", 4)
print_shoping_list(shoping_list)
