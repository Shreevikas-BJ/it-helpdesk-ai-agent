# app/nodes.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Literal, Tuple
import json, os, torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional deps (handled gracefully)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore

try:
    from langchain_community.llms import Ollama  # type: ignore
except Exception:
    Ollama = None  # type: ignore

from .retriever import load_kb
from .tools import safe_shell
from .judge import judge
from .config import CLASSIFIER_CKPT, TOP_K, TOP_K_RERANK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class S(TypedDict):
    ticket_text: str
    category: str | None
    confidence: float | None
    candidates: List[str]
    evidence: List[str]
    plan: str | None
    commands: List[str]
    execution: List[Dict]
    verdict: Literal["PASS", "FIX"] | None
    final: str | None
    log: List[Dict]


# -----------------------------
# Lazy singletons (Streamlit-safe)
# -----------------------------
_label_maps: Dict | None = None
_id2label: Dict[int, str] | None = None
_tok = None
_cls = None
_kb = None
_ce = None


def _load_classifier_once() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Dict[int, str]]:
    """Load tokenizer/model/labels exactly once per process."""
    global _label_maps, _id2label, _tok, _cls
    if _tok is None or _cls is None or _id2label is None:
        labels_path = os.path.join(CLASSIFIER_CKPT, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            _label_maps = json.load(f)
        _id2label = {int(k): v for k, v in _label_maps["id2label"].items()}
        _tok = AutoTokenizer.from_pretrained(CLASSIFIER_CKPT)
        _cls = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_CKPT).to(DEVICE).eval()
    return _tok, _cls, _id2label  # type: ignore


def _load_retrieval_once():
    """Load KB and cross-encoder reranker once."""
    global _kb, _ce
    if _kb is None:
        _kb = load_kb()
    if _ce is None:
        if CrossEncoder is None:
            raise RuntimeError("sentence_transformers not available. Install it or disable reranking.")
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _kb, _ce


# -----------------------------
# Nodes
# -----------------------------
def classify_node(s: S) -> S:
    tok, cls_model, id2label = _load_classifier_once()
    enc = tok(s["ticket_text"], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(**enc).logits
        pred = int(logits.argmax(dim=-1)[0])
        conf = float(torch.softmax(logits, dim=-1)[0][pred])
    s["category"] = id2label[pred]
    s["confidence"] = conf
    s.setdefault("log", []).append({"node": "classify", "category": s["category"], "conf": round(conf, 3)})
    return s


def retrieve_node(s: S) -> S:
    kb, _ = _load_retrieval_once()
    docs = kb.similarity_search(s["ticket_text"], k=TOP_K)
    s["candidates"] = [d.page_content for d in docs]
    s.setdefault("log", []).append({"node": "retrieve", "k": len(s["candidates"])})
    return s


def rerank_node(s: S) -> S:
    _, ce = _load_retrieval_once()
    pairs = [(s["ticket_text"], d) for d in s["candidates"]]
    scores = ce.predict(pairs).tolist()
    ranked = sorted(zip(s["candidates"], scores), key=lambda x: x[1], reverse=True)[:TOP_K_RERANK]
    s["evidence"] = [d for d, _ in ranked]
    s.setdefault("log", []).append({"node": "rerank", "kept": len(s["evidence"])})
    return s


# Template planner (LLM-free)
TEMPLATES = {
    "vpn_issue": (
        "1) Check internet\n2) Ping 8.8.8.8 (3 packets)\n3) Update VPN\n4) Try wired/stable Wi-Fi\n5) Reboot\n6) Escalate with logs",
        ["ping 8.8.8.8 -n 3", "whoami"],
    ),
    "password_reset": (
        "1) Confirm username\n2) Self-service reset\n3) Wait 15 mins if locked\n4) Update on all devices\n5) Resync MFA\n6) Escalate if portal fails",
        ["echo Password reset steps shared"],
    ),
    "email_bounce": (
        "1) Verify recipient address\n2) Check quota\n3) Plain-text test\n4) Check spam/junk\n5) Collect bounce code",
        ["echo Collected bounce code"],
    ),
    "wifi_dropout": (
        "1) Move closer to AP\n2) Forget & reconnect SSID\n3) Disable NIC power saving\n4) Ping gateway\n5) Update drivers",
        ["ping 8.8.8.8 -n 3"],
    ),
    "software_install": (
        "1) Verify license/approval\n2) Use company portal\n3) Admin install if allowed\n4) Document version/purpose",
        ["whoami"],
    ),
    "account_lock": (
        "1) Confirm UPN\n2) Check CAPS LOCK\n3) Backup MFA\n4) Resync authenticator time\n5) Follow ID verification",
        ["echo MFA resync guidance shared"],
    ),
    "printer_error": (
        "1) Power cycle\n2) Check network\n3) Clear spooler\n4) Reinstall driver\n5) Local test page",
        ["echo Clear spooler step documented"],
    ),
}


def propose_fix_node(s: S) -> S:
    # If classifier confidence is high → use template (only if category matches)
    if s.get("confidence", 0) > 0.85 and s.get("category") in TEMPLATES:
        plan, cmds = TEMPLATES[s["category"]]  # type: ignore[index]
        s["plan"] = plan
        s["commands"] = cmds
        s.setdefault("log", []).append({"node": "propose_fix", "mode": "template", "commands": cmds})
        return s

    # Otherwise → ask Ollama (Mistral default) if available
    context = "\n\n".join(s.get("evidence", [])) or "No KB evidence found."
    prompt = f"""
You are an IT helpdesk assistant.
Based on the following ticket and KB evidence, propose step-by-step troubleshooting steps
and 1-2 safe Windows diagnostic commands.

Ticket: {s['ticket_text']}
Evidence: {context}

Rules:
- Commands must be safe only (ping, ipconfig, whoami, echo).
- Return your answer in JSON with keys "plan" (string) and "commands" (list).
""".strip()

    used_mode = "ollama"
    raw = None
    if Ollama is not None:
        try:
            llm = Ollama(model="mistral")
            raw = llm.invoke(prompt)
        except Exception:
            used_mode = "ollama-error"

    if raw:
        try:
            parsed = json.loads(raw)
            s["plan"] = parsed.get("plan", "")
            s["commands"] = parsed.get("commands", [])
            s.setdefault("log", []).append({"node": "propose_fix", "mode": used_mode, "commands": s["commands"]})
            return s
        except Exception:
            used_mode = "ollama-fallback"

    # Fallback if Ollama unavailable or parsing failed
    s["plan"] = "1) Collect basic network/system info\n2) Try safe diagnostics (ping/whoami)\n3) Cross-check KB\n4) Escalate with logs"
    s["commands"] = ["whoami"]
    s.setdefault("log", []).append({"node": "propose_fix", "mode": used_mode})
    return s


def judge_node(s: S) -> S:
    j = judge(s.get("plan") or "", s.get("commands") or [])
    s["verdict"] = j.get("verdict") or "FIX"
    s.setdefault("log", []).append({"node": "judge", "verdict": s["verdict"], "why": j.get("explanation", "")})
    return s


def execute_node(s: S) -> S:
    if s.get("verdict") != "PASS":
        s["execution"] = [{"ok": False, "stderr": "blocked by judge"}]
        s["final"] = "Plan requires fix. Not executed."
        return s
    results = []
    for c in s.get("commands", []):
        results.append(safe_shell(c))
    s["execution"] = results
    s["final"] = f"Executed {len(results)} commands. See logs."
    s.setdefault("log", []).append({"node": "execute", "ran": len(results)})
    return s
