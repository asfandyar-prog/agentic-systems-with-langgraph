# email-triage-workflow

A customer-support email workflow built from the LangGraph "Thinking in LangGraph" tutorial. An LLM classifies the email and code routes it through stub documentation-search or bug-ticket steps. The LLM then drafts a reply, and billing, high-urgency or complex emails pause for human approval before a stub send.

## Agentic or not

Not agentic. It is a fixed workflow with two LLM calls, no tools and no loop. Cell numbers are 0-based.

- Classification: `with_structured_output(EmailClassification)` (cell 4, line 5). Routing is an `if/elif` on the result (cell 4, lines 21-29), returned as `Command(goto=...)`.
- Documentation search and bug tracking return hardcoded strings (cells 5 and 6).
- Human review: `interrupt(...)` (cell 7, line 56) with a `MemorySaver` checkpointer (cell 8, lines 28-29), resumed with `Command(resume=...)` (cell 9, line 18).
- `send_reply` only prints (cell 7, line 78).

## Run

```
uv sync
uv pip install ipykernel
```

Open [email-triage-workflow.ipynb](email-triage-workflow.ipynb) with the repo's `.venv` as the kernel. Put `GROQ_API_KEY` in a `.env` in this folder (git-ignored); without it, cell 2 raises `TypeError` at line 10. Run cells 1-9 in order.

Verified offline: cells 1-8 build the graph. The saved output of cell 9 shows a complete run: it pauses at `human_review`, resumes with an edited reply, and `send_reply` prints it. In that run the LLM classified a double-charge email as `bug`, not `billing`.

## Not finished or not working

- Emails classified as `question` or `feature` crash. Cell 5, line 4 reads `state['classification',{}]`, which raises `TypeError: unhashable type: 'dict'`.
- Cell 5, line 17 catches `SearchAPIError`, which is never defined.
- `bug_tracking` writes `current_step`, which is not a state field (cell 6, line 7).
- No real email, documentation or ticketing integration.

Credits: LangChain documentation, "Thinking in LangGraph" (customer support email agent).
