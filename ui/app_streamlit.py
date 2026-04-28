# ui/app_streamlit.py
import os, sys, traceback
import streamlit as st

# --- Ensure project root is on sys.path so "from app.graph import app" works ---
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the graph (fast) and the lazy-loading nodes (for warmup)
from app.graph import app
from app import nodes

# ---------- Page config + styling ----------
st.set_page_config(page_title="IT Helpdesk Agent", page_icon="🛠", layout="centered")
st.markdown("""
<style>
.block-container { max-width: 900px; }
.plan-wrap {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.95rem;
  background: rgba(0,0,0,0.03);
  border: 1px solid rgba(49,51,63,0.2);
  padding: 0.75rem 0.9rem;
  border-radius: 10px;
  margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("IT Helpdesk Agent (Local • Free)")
st.caption("Classify → Retrieve → Rerank → Plan (Template/Ollama) → Judge → Execute (safe commands only)")

# ---------- Helpers ----------
def warmup_models() -> bool:
    with st.spinner("Loading classifier / KB / reranker (first run may download models)…"):
        # Classifier once
        try:
            nodes._load_classifier_once()
        except Exception as e:
            st.error(f"Classifier load failed: {e}")
            return False

        # KB + Cross-Encoder once
        try:
            nodes._load_retrieval_once()
        except RuntimeError as e:
            # e.g., sentence_transformers missing or disabled
            st.warning(f"Reranker unavailable: {e}\nRetrieval will still work without rerank.")
        except Exception as e:
            st.error(f"KB/reranker load failed: {e}")
            return False

    st.success("Models ready ✔")
    return True

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    if st.button("Warm up models"):
        warmup_models()
    st.write("Tip: First run may be slow while models download.\nUse **Warm up models** to see progress.")

# ---------- Input ----------
txt = st.text_area(
    "Paste a helpdesk ticket:",
    value="",
    height=140,
    placeholder='e.g., "VPN disconnects every 10 minutes on Windows 11."',
)

run = st.button("Run Agent", type="primary")

# ---------- Run pipeline ----------
if run and txt.strip():
    # Safe to warm up before running
    ok = warmup_models()
    if not ok:
        st.stop()

    state = {
        "ticket_text": txt.strip(),
        "category": None, "confidence": None,
        "candidates": [], "evidence": [],
        "plan": None, "commands": [],
        "execution": [], "verdict": None,
        "final": None, "log": []
    }

    try:
        with st.spinner("Running agent…"):
            out = app.invoke(state)
    except Exception:
        st.error("Pipeline error while invoking the graph.")
        st.code(traceback.format_exc())
        st.stop()

    # ---------- Results ----------
    st.subheader("Result")

    conf = out.get("confidence")
    conf_str = f"{conf:.2f}" if conf is not None else "—"
    st.write(f"**Category:** {out.get('category', '—')} (conf {conf_str})")

    st.write("**Plan:**")
    safe_plan = (out.get("plan") or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='plan-wrap'>{safe_plan}</div>", unsafe_allow_html=True)

    st.write("**Commands:**")
    st.text_area(
        "Commands",
        value="\n".join(out.get("commands") or []),
        height=120,
        disabled=True,
        label_visibility="collapsed"
    )

    st.write("**Verdict:**", out.get("verdict", "—"))
    with st.expander("Execution logs"):
        st.json(out.get("execution") or [])
    with st.expander("Trace log"):
        st.json(out.get("log") or [])

elif run and not txt.strip():
    st.warning("Please paste a ticket description before running the agent.")
