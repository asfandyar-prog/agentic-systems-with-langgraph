# agentic-systems-with-langgraph: mostly learning work

This repository is mostly learning work from studying LangGraph: course exercises, a documentation tutorial, and prototypes. It has one real project, a Streamlit app that turns a topic into a Markdown article through a fixed LangGraph pipeline, plus an earlier variant of that app. Despite the repository name, almost none of the code is agentic. Only one notebook cell builds a loop in which the model picks tools and decides when to stop, and that loop has never been given a real question.

Text generation uses Groq through `langchain-groq` (default model `llama-3.3-70b-versatile`), and the content pipelines request images from Gemini through `google-genai`. Python 3.13 ([.python-version](.python-version)), `langgraph>=1.0.10` ([pyproject.toml](pyproject.toml)).

## Contents

| Project | What it does | Model calls tools? | Model controls the loop? | Status |
|---|---|---|---|---|
| [projects/content-pipeline](projects/content-pipeline/) | Topic, then optional Tavily search, LLM plan, sections written one at a time, and a Markdown file with Gemini images. Streamlit UI | No. Code calls Tavily ([backend.py:252](projects/content-pipeline/backend.py#L252)) | No. Code stops the worker loop after the last planned task ([backend.py:448](projects/content-pipeline/backend.py#L448)) | partial |
| [projects/content-pipeline-parallel](projects/content-pipeline-parallel/) | Earlier variant of the same pipeline that writes sections in parallel with `Send` ([backend.py:296](projects/content-pipeline-parallel/backend.py#L296)) | No | No loop | partial |
| [learning/orchestrator-worker-blog](learning/orchestrator-worker-blog/) | First prototype: LLM plans sections, parallel writers, joined Markdown | No | No loop | partial: fails as saved |
| [learning/langgraph-exercises](learning/langgraph-exercises/) | Course exercises: small graphs, Groq chatbot, ReAct agent with an `add` tool, unfinished Drafter agent | Yes, in the ReAct cell (02 cell 10, line 52) and in the broken Drafter cells | Yes, in the ReAct cell (lines 61-65), never given a real question | partial |
| [learning/email-triage-workflow](learning/email-triage-workflow/) | Docs tutorial: LLM classifies an email, code routes with `Command`, LLM drafts a reply, `interrupt` waits for human approval | No. Search and ticketing are stubs | No loop | partial |
| [archive](archive/) | Scratch, duplicates, broken experiments, generated outputs | n/a | n/a | archived |

"partial" means the code builds but has known defects or has not been verified end to end. Each folder's README lists them with file and line references. Notebook cell numbers are 0-based.

## Setup

```
uv sync
uv pip install -r projects/content-pipeline/requirements.txt   # Streamlit apps
uv pip install ipykernel                                        # notebooks
```

`uv sync` removes packages that are not in `uv.lock`, and streamlit, pandas, google-genai and ipykernel are not in it. Re-run the `uv pip install` lines after every `uv sync`, or add those packages to `pyproject.toml`. `pyproject.toml` also lists `langchainhub`, which no code imports.

API keys go in a `.env` in the folder of the code that uses them. `.env` is git-ignored everywhere ([.gitignore](.gitignore)).

## What is not here

- No reflection or self-critique loops.
- No long-term memory. Outside `archive/`, the only checkpointer is `MemorySaver` in the email notebook.
- No multi-agent system. The content pipeline's router, planner and workers are nodes in one fixed graph.
- No tests.

## Archive

| Path | Why it is archived |
|---|---|
| [archive/basic-chatbot](archive/basic-chatbot/) | `1-basicchatbot.ipynb` is one LLM call, the same as the first exercise bot. `2-react-agent.ipynb` never defines `tools` or `llm_with_tool` and has a saved `NameError` |
| [archive/practice-scripts](archive/practice-scripts/) | `01-practice.py` crashes at line 16. `02-practice.ipynb` returns the wrong state key (`"Messages"`). `sub-graph.py` crashes at line 81 |
| [archive/agent-executor-scratch](archive/agent-executor-scratch/) | `01-agent-exicutor.py` uses LangChain's legacy `AgentExecutor` (`ImportError` on langchain 1.x). `agent_executor.ipynb` has one failing cell. `agent-4` is a broken copy of the exercise chatbot (`NameError` at line 34) |
| [archive/duplicates](archive/duplicates/) | `helloword.ipynb` duplicates the first 10 cells of `01-graph-basics.ipynb`. `research-assitent-blog-writing.ipynb` is an older copy of the orchestrator-worker notebook |
| [archive/generated-outputs](archive/generated-outputs/) | Articles written by the pipelines, plus a chatbot log. Every `*_orchestrated.md` contains three failed image blocks (Gemini error 429) |
| [archive/main.py](archive/main.py) | `uv init` placeholder |

## License

MIT, see [LICENSE](LICENSE).
