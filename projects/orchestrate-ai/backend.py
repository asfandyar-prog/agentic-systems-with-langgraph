
from __future__ import annotations

import os
import re
import time
import operator
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    title: str
    audience: str
    tone: str
    content_type: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
        "report",
        "whitepaper",
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. diagram.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # sequential workers
    current_task_index: int
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# -----------------------------
# 2) LLM
# -----------------------------
from langchain_groq import ChatGroq  # noqa: E402

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

# -----------------------------
# 3) Retry wrapper (rate-limit safe)
# -----------------------------
def _is_rate_limit_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("rate limit" in s) or ("rate_limit" in s) or ("tpm" in s) or ("429" in s)


def _backoff_sleep(attempt: int) -> None:
    # Exponential-ish backoff: 1.0s, 1.8s, 3.2s, 5.6s (+ tiny jitter)
    base = 1.0 * (1.8 ** attempt)
    jitter = 0.05 * attempt
    time.sleep(min(8.0, base + jitter))


def safe_invoke(messages, structured_model=None):
    """
    One entry-point for all LLM calls.
    - Retries rate-limit errors with backoff
    - Leaves non-rate errors to raise immediately (so you see real bugs)
    """
    max_attempts = int(os.getenv("LLM_MAX_RETRIES", "5"))
    last_err: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            if structured_model is not None:
                return llm.with_structured_output(structured_model).invoke(messages)
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) and attempt < max_attempts - 1:
                _backoff_sleep(attempt)
                continue
            raise
    raise last_err or RuntimeError("LLM call failed.")


# -----------------------------
# 4) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical content orchestration system.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book, ensure queries reflect the last 7 days relative to As-of date.
"""

def router_node(state: State) -> dict:
    decision: RouterDecision = safe_invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        structured_model=RouterDecision,
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries[:10],
        "recency_days": recency_days,
        "current_task_index": 0,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 5) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    pack: EvidencePack = safe_invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ],
        structured_model=EvidencePack,
    )

    dedup: Dict[str, EvidenceItem] = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # For open_book, hard recency filter
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    # Token discipline: cap evidence in state
    max_evidence = int(os.getenv("MAX_EVIDENCE_ITEMS", "12"))
    return {"evidence": evidence[:max_evidence]}


# -----------------------------
# 6) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical content architect.
Produce a highly actionable plan for a technical content artifact.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Keep tasks ordered for best reader flow.
- Tags are optional.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set content_type="news_roundup"
  - Do NOT include tutorial content unless explicitly requested.
  - If evidence is weak, the plan must reflect that (don’t invent events).
  - Prefer “what happened + implications”.

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", []) or []

    # Deterministic guard: open_book with weak evidence → downgrade to hybrid.
    if mode == "open_book" and len(evidence) < 2:
        mode = "hybrid"

    forced_type = "news_roundup" if mode == "open_book" else None

    # Token discipline: compact evidence sent to planner
    ev_compact = [
        {
            "title": e.title[:140],
            "url": e.url,
            "published_at": e.published_at,
            "source": e.source,
        }
        for e in evidence[:10]
    ]

    plan: Plan = safe_invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force content_type=news_roundup' if forced_type else ''}\n\n"
                    f"Evidence (compact):\n{ev_compact}"
                )
            ),
        ],
        structured_model=Plan,
    )

    if forced_type:
        plan.content_type = "news_roundup"

    return {"plan": plan, "mode": mode, "current_task_index": 0}


# -----------------------------
# 7) Worker (SEQUENTIAL)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical content architect.
Write ONE section of a technical artifact in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If content_type=="news_roundup", do NOT drift into tutorials.
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

_URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)

def _scrub_unverified_urls(text: str, allowed_urls: set[str]) -> str:
    found = set(_URL_RE.findall(text or ""))
    for url in found:
        if url not in allowed_urls:
            text = text.replace(url, "UNVERIFIED_SOURCE")
    return text

def worker_node(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("worker_node called without plan")

    idx = int(state.get("current_task_index", 0))
    if idx >= len(plan.tasks):
        # nothing left; let router decide next
        return {}

    task = plan.tasks[idx]
    evidence = state.get("evidence", []) or []
    mode = state.get("mode", "closed_book")

    # Token discipline: send only top N evidence lines to worker
    ev_lines = []
    for e in evidence[:8]:
        ev_lines.append(f"- {e.title[:140]} | {e.url} | {e.published_at or 'date:unknown'}")
    evidence_text = "\n".join(ev_lines) if ev_lines else "(none)"

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    section_md = safe_invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Title: {plan.title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Content type: {plan.content_type}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state.get('as_of')} (recency_days={state.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # Deterministic hygiene for open_book: no unverified URLs.
    if mode == "open_book":
        allowed = {e.url for e in evidence if e.url}
        section_md = _scrub_unverified_urls(section_md, allowed)

    return {
        "sections": [(task.id, section_md)],
        "current_task_index": idx + 1,
    }

def next_after_worker(state: State) -> str:
    plan = state.get("plan")
    if plan is None:
        return "reducer"
    idx = int(state.get("current_task_index", 0))
    return "worker" if idx < len(plan.tasks) else "reducer"


# -----------------------------
# 8) Reducer with images (subgraph)
# -----------------------------
def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "artifact"

def merge_content(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("merge_content called without plan.")

    ordered_sections = [md for _, md in sorted(state.get("sections", []), key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.title}\n\n{body}\n"
    return {"merged_md": merged_md}

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS artifact.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    plan = state.get("plan")
    assert plan is not None

    image_plan: GlobalImagePlan = safe_invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Content type: {plan.content_type}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{state.get('merged_md','')}"
                )
            ),
        ],
        structured_model=GlobalImagePlan,
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")

def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")
    assert plan is not None

    md = state.get("md_with_placeholders") or state.get("merged_md") or ""
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.title)}_orchestrated.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.title)}_orchestrated.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


# Reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# -----------------------------
# 9) Build main graph (SEQUENTIAL)
# -----------------------------
g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

# Sequential execution:
g.add_edge("orchestrator", "worker")
g.add_conditional_edges("worker", next_after_worker, {"worker": "worker", "reducer": "reducer"})

g.add_edge("reducer", END)

orchestrate_app = g.compile()

# Backward-compatible alias (so old frontend import still works)
app = orchestrate_app