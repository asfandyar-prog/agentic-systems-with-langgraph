# orchestrator-worker-blog

First prototype of [projects/content-pipeline](../../projects/content-pipeline/). One LLM call plans a blog post as structured sections (5-7 per the prompt). LangGraph `Send` then starts one writer per section in parallel, and a reducer joins the sections and saves `<title>.md`.

## Agentic or not

Not agentic. It is a fixed graph, `orchestrator -> worker x N -> reducer`, with no tools and no loop. The LLM decides only how many sections there are; code fans them out and ends the run. Cell numbers are 0-based.

- Plan: `llm.with_structured_output(Plan)` (cell 5, line 3)
- Fan-out: `Send("worker", ...)` (cell 6, line 2), wired in cell 10 (line 2)
- Save: `Path(filename).write_text(...)` (cell 8, line 13)

## Run

```
uv sync
uv pip install ipykernel
```

Open [orchestrator-worker-blog.ipynb](orchestrator-worker-blog.ipynb) with the repo's `.venv` as the kernel. Put `GROQ_API_KEY` in a `.env` in this folder (git-ignored) and run all cells. It fails as saved; see below.

## Not finished or not working

- Cell 7, line 12: `section_md = llm.invoke(` is indented five spaces instead of four, so the cell raises `IndentationError`.
- Cell 1, line 18 uses `Literal`, but cell 0 never imports it (line 4). Building the schema fails with `NameError: name 'Literal' is not defined`.
- The saved output in cell 12 came from an earlier version of these cells.
- No research, citations, images or UI. Those exist only in the `projects/` pipelines.

Credits: Asfand Yar.
