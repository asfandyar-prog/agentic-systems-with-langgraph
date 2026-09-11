# content-pipeline-parallel

An earlier variant of [content-pipeline](../content-pipeline/): a Streamlit app that runs the same topic, research, plan, sections and images pipeline and saves a Markdown blog post. The difference is that it writes all sections in parallel with LangGraph `Send` ([backend.py:296-312](backend.py#L296-L312)), with no retry wrapper, evidence caps or URL scrubbing.

## Agentic or not

Not agentic. It is a fixed graph with LLM calls and no loop: each planned task gets one `worker`, then the reducer runs once ([backend.py:544-559](backend.py#L544-L559)). No tools are bound to the model. Code calls Tavily ([backend.py:212-213](backend.py#L212-L213)), and the LLM calls return fixed Pydantic schemas ([backend.py:139](backend.py#L139)).

## Run

```
uv sync
uv pip install -r projects/content-pipeline/requirements.txt
cd projects/content-pipeline-parallel
uv run streamlit run frontend.py
```

Run from the repo root. Dependencies are the same as content-pipeline, and this folder has no requirements file. `GROQ_API_KEY` must be set, for example in a git-ignored `.env` here, before the app starts. Otherwise importing `backend.py` raises `TypeError: str expected, not NoneType` ([backend.py:119](backend.py#L119)). `TAVILY_API_KEY` and `GOOGLE_API_KEY` are optional. Output goes to the current directory: `<title>.md` and `images/` ([backend.py:495-500](backend.py#L495-L500)).

Verified: with a key set, the graph compiles and the app's first render raises no exceptions (Streamlit AppTest). Without a key it fails as described above. A full run with keys was not verified for this README.

## Not finished or not working

- All workers call Groq at the same time with no rate-limit handling. The file has no retry code, so any Groq error fails the run.
- The "Past blogs" sidebar lists every `*.md` in the current directory, not only generated posts ([frontend.py:150-158](frontend.py#L150-L158)).
- Tavily errors are swallowed ([backend.py:186-187](backend.py#L186-L187)), and `TavilySearchResults` is marked deprecated by the installed LangChain.
- A leftover notebook expression `app` is on the last line ([backend.py:560](backend.py#L560)).
- No tests.

Credits: Asfand Yar.
