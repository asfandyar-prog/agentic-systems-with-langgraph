from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

from backend import orchestrate_app as app


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="OrchestrateAI – Content Engine",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# STYLING (Clean + Professional)
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .metric-card {
        background-color: #111;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #222;
        border: 1px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
## OrchestrateAI
**Deterministic Multi-Agent Content Orchestration System**
Built with LangGraph
""")

st.title("🧠 OrchestrateAI")
st.caption("Multi-Agent Content Orchestration Engine powered by LangGraph")

st.divider()

# ============================================================
# HELPERS
# ============================================================

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "artifact"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        yield ("final", graph_app.invoke(inputs))
        return
    except Exception:
        pass

    yield ("final", graph_app.invoke(inputs))


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Execution Control")

    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())

    run_btn = st.button("🚀 Execute Orchestration", type="primary")

    st.divider()
    st.markdown("### 📊 System Info")
    st.caption("Router → Research → Orchestrator → Workers → Reducer")

# ============================================================
# SESSION STATE
# ============================================================
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ============================================================
# EXECUTION
# ============================================================
if run_btn:
    if not topic.strip():
        st.warning("Enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Running OrchestrateAI...", expanded=True)
    progress = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind == "updates":
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1:
                node_name = next(iter(payload.keys()))

            if node_name and node_name != last_node:
                status.write(f"➡️ {node_name}")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "evidence": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", []))
                if isinstance(current_state.get("plan"), dict)
                else None,
                "sections": len(current_state.get("sections", []) or []),
                "images": len(current_state.get("image_specs", []) or []),
            }
            progress.json(summary)

        elif kind == "final":
            st.session_state["last_out"] = payload
            status.update(label="✅ Completed", state="complete")

# ============================================================
# DISPLAY OUTPUT
# ============================================================
out = st.session_state.get("last_out")

if not out:
    st.info("Run the system to generate an artifact.")
    st.stop()

plan_obj = out.get("plan")

# ============================================================
# TOP METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Mode", out.get("mode"))
col2.metric("Evidence", len(out.get("evidence") or []))
col3.metric("Sections", len(out.get("sections") or []))
col4.metric("Images", len(out.get("image_specs") or []))

st.divider()

# ============================================================
# TABS
# ============================================================
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Output", "🖼️ Images", "🧾 Logs"]
)

# ---------------- PLAN ----------------
with tab_plan:
    if not plan_obj:
        st.info("No plan available.")
    else:
        plan = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj

        st.subheader(plan.get("title"))
        st.write(f"**Audience:** {plan.get('audience')}")
        st.write(f"**Tone:** {plan.get('tone')}")
        st.write(f"**Content Type:** {plan.get('content_type')}")

        tasks = plan.get("tasks", [])
        if tasks:
            df = pd.DataFrame(tasks)
            st.dataframe(df, width="stretch")

# ---------------- EVIDENCE ----------------
with tab_evidence:
    evidence = out.get("evidence") or []
    if not evidence:
        st.info("No evidence used.")
    else:
        for e in evidence:
            e = e.model_dump() if hasattr(e, "model_dump") else e
            with st.container():
                st.markdown(f"### {e.get('title')}")
                st.write(e.get("url"))
                st.caption(f"{e.get('source')} · {e.get('published_at')}")
                st.divider()

# ---------------- PREVIEW ----------------
with tab_preview:
    final_md = out.get("final") or ""
    if not final_md:
        st.warning("No final output.")
    else:
        st.markdown(final_md, unsafe_allow_html=False)

        title = (
            plan_obj.title
            if hasattr(plan_obj, "title")
            else plan_obj.get("title", "artifact")
        )

        filename = f"{safe_slug(title)}_orchestrated.md"

        st.download_button(
            "⬇️ Download Markdown",
            data=final_md.encode("utf-8"),
            file_name=filename,
            mime="text/markdown",
        )

        bundle = bundle_zip(final_md, filename, Path("images"))
        st.download_button(
            "📦 Download Bundle",
            data=bundle,
            file_name=f"{safe_slug(title)}_bundle.zip",
            mime="application/zip",
        )

# ---------------- IMAGES ----------------
with tab_images:
    specs = out.get("image_specs") or []
    images_dir = Path("images")

    if specs:
        st.json(specs)

    if images_dir.exists():
        files = [p for p in images_dir.iterdir() if p.is_file()]
        for p in files:
            st.image(str(p), caption=p.name, use_container_width=True)

# ---------------- LOGS ----------------
with tab_logs:
    st.text_area(
        "Execution Log",
        value=json.dumps(out, indent=2, default=str),
        height=500,
    )