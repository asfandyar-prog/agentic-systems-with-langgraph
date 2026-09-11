# langgraph-exercises

Two notebooks of beginner LangGraph course exercises. [01-graph-basics.ipynb](01-graph-basics.ipynb) builds small graphs with plain Python nodes and no LLM. [02-chatbot-react-drafter.ipynb](02-chatbot-react-drafter.ipynb) adds a Groq chatbot, a ReAct agent with one `add` tool, and an unfinished "Drafter" document agent.

Cell numbers below are 0-based positions and include markdown cells.

## Agentic or not

Only 02 cell 10 is agentic in the strict sense. The model gets a tool via `.bind_tools(tools)` (line 52) and a `ToolNode` (line 72), and `should_continue` ends the loop when the model's reply has no `tool_calls` (lines 61-65). That loop has never been given a real question (see below).

Everything else is fixed control flow. The loop in 01 cell 27 stops when a counter reaches 4 (line 25). The chatbot in 02 cells 2-3 loops in plain Python until the user types `exit` (cell 3, line 3). The Drafter cells bind tools but are wired incorrectly.

## Run

```
uv sync
uv pip install ipykernel
```

`ipykernel` is not in `uv.lock`, so `uv sync` removes it. Open a notebook in VS Code or Jupyter with the repo's `.venv` as the kernel. 01 needs no API key. 02 needs `GROQ_API_KEY` in a `.env` in this folder (git-ignored), and its cells 1, 3 and 12 read from `input()`.

Verified offline: 01 cells 0-27 execute, and 02 cell 10 builds its graph.

## Not finished or not working

- 01 cell 21: the router reads `state["operator"]` and `state["opeartion"]` (lines 25, 27), but the field is `operation`. Never invoked.
- 01 cell 22: `router_1` returns `"subtractor_operation_2"` (line 25), but the edge map expects `"subtractor_operation"` (line 65). Never invoked.
- 01 cell 32 raises `TypeError` at line 17 (`Sequence[BaseMessage,add_messages]`). 02 cell 12 repeats the cell with that line fixed.
- 02 cell 11: the ReAct test sends `{'message': [{"user","Add 43 + 42"}]}` (line 9), which has the wrong key and a set instead of a message. The model gets no question, and the saved reply is "What is your query?". No saved output shows a tool call.
- 02 cell 12 (Drafter): the `agent -> tools` edge is unconditional (line 120). Routing uses cell 10's `should_continue` (line 123), because this cell defines its router as `should_countinue` (line 86).
- 02 cell 13 calls `print_messages`, but the function is named `print_message` (cell 12, line 102), and nothing calls `run_document_agent`. Cell 14 is a syntax error.

Credits: freeCodeCamp, "LangGraph Complete Course for Beginners" by Vaibhav Mehra.
