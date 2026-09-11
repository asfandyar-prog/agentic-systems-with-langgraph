# content-pipeline

A Streamlit app that takes a topic and an as-of date and writes a Markdown article to disk. A fixed LangGraph pipeline decides whether web research is needed, optionally searches with Tavily, plans the sections, writes them one at a time with Groq, and asks Gemini for diagrams.

## Agentic or not

Not agentic: this is a fixed pipeline with LLM calls. The model calls no tools. Code calls Tavily for every search query the router produced ([backend.py:252-253](backend.py#L252-L253)), and each LLM call returns a fixed Pydantic schema ([backend.py:148](backend.py#L148)). Code ends the worker loop once every planned task is written ([backend.py:448-453](backend.py#L448-L453)). The model only decides whether to research ([backend.py:200-201](backend.py#L200-L201)) and how many tasks to plan.

Graph: `router -> research (optional) -> orchestrator -> worker (repeats) -> reducer`. The reducer is a subgraph `merge_content -> decide_images -> generate_and_place_images` ([backend.py:601-633](backend.py#L601-L633)).

## Run

```
uv sync
uv pip install -r projects/content-pipeline/requirements.txt
cd projects/content-pipeline
uv run streamlit run frontend.py
```

Run from the repo root. `uv sync` removes streamlit, pandas and google-genai because they are not in `pyproject.toml`, so install the requirements after it. Put a `.env` in this folder (git-ignored):
- `GROQ_API_KEY`: required. `GROQ_MODEL` overrides the default `llama-3.3-70b-versatile` ([backend.py:119](backend.py#L119)).
- `TAVILY_API_KEY`: optional. Without it, research returns no evidence ([backend.py:208-209](backend.py#L208-L209)).
- `GOOGLE_API_KEY`: optional. Without it, each image becomes a failure note in the Markdown ([backend.py:582-590](backend.py#L582-L590)).

Output goes to the current directory: `<title>_orchestrated.md` and `images/` ([backend.py:570](backend.py#L570), [backend.py:595](backend.py#L595)). Verified: the graph compiles, and the app's first render raises no exceptions (Streamlit AppTest, no key). A full run with keys was not verified for this README.

## Not finished or not working

- The final state's `sections` list is doubled. The reducer subgraph ([backend.py:621](backend.py#L621)) returns its whole state, and the `operator.add` reducer on `sections` ([backend.py:104](backend.py#L104)) appends the list again. The article is unaffected, but the app's "Sections" metric shows twice the real count ([frontend.py:209](frontend.py#L209)).
- Image generation has not succeeded in any saved output. All 7 `*_orchestrated.md` files in [archive/generated-outputs](../../archive/generated-outputs/) contain three "IMAGE GENERATION FAILED" blocks with Gemini error 429.
- URLs are checked in code only in `open_book` mode ([backend.py:439-441](backend.py#L439-L441)). In other modes the citation rules exist only in the prompt. [self-supervised_learning_orchestrated.md](../../archive/generated-outputs/self-supervised_learning_orchestrated.md), line 3, cites `https://www.example.com`.
- Tavily errors are swallowed ([backend.py:226-227](backend.py#L226-L227)). The `TavilySearchResults` class it imports ([backend.py:211](backend.py#L211)) is marked deprecated by the installed LangChain.
- The 5-9 task limit exists only in the prompt ([backend.py:296](backend.py#L296)). `Plan.tasks` has no bound ([backend.py:50](backend.py#L50)).
- No tests.

Credits: Asfand Yar. This is the later variant of [content-pipeline-parallel](../content-pipeline-parallel/); see the compatibility alias at [backend.py:635-636](backend.py#L635-L636).
