# app/judge.py
import re
from typing import List, Dict

# ---- Strict allowlist: EXACT first token must be one of these ----
# (We keep this very small on purpose. Add more only if you are sure.)
SAFE_BINARIES = {
    "ping",
    "whoami",
    "echo",
    "ipconfig",
}

# ---- Common dangerous patterns to block regardless of binary ----
# (Lowercased matching; keep this conservative but useful)
BLACKLIST_PATTERNS = [
    r"\bformat\s+[a-z]:",                          # format C:
    r"\brm\s+-rf\b",                               # rm -rf (posix)
    r"\brmdir\s+/s\s+/q\b",                        # rmdir /s /q
    r"\brd\s+/s\s+/q\b",                           # rd /s /q
    r"\bdel\s+/?[pswaPSWA]*\s+\*\.?\*?\b",         # del *.* variations
    r"\bnetsh\s+advfirewall\s+set\s+allprofiles\s+state\s+off\b",
    r"\bshutdown\b",                               # shutdown /s /r /t
    r"\bsc\s+stop\b",                              # sc stop <service>
    r"\btaskkill\s+/f\b",                          # taskkill /F
    r"\bnet\s+user\b.*(/add|/delete|/active)",     # net user admin ops
    r"\bbcde?dit\b",                               # bcdedit
    r"\breg\s+delete\b",                           # registry delete
    r"\bwmic\b.*(delete|terminate|process)",       # wmic dangerous ops
    r"\bpowershell\b.*set-mppreference.*disablerealtimemonitoring",  # Defender off
]

BLACKLIST_RE = [re.compile(p, re.I) for p in BLACKLIST_PATTERNS]


def _first_token(cmd: str) -> str:
    return (cmd or "").strip().split()[0].lower() if cmd.strip() else ""


def _find_blacklist_hit(cmd: str) -> str | None:
    s = cmd.lower()
    for rx in BLACKLIST_RE:
        if rx.search(s):
            return rx.pattern
    return None


def judge(plan: str, commands: List[str]) -> Dict[str, str]:
    """
    Decide if it's safe to execute the proposed commands.
    Returns:
        { "verdict": "PASS" | "FIX", "explanation": "<why>" }
    Policy:
      1) Command list must be non-empty.
      2) Each command's *first token* must be in SAFE_BINARIES.
      3) If any command matches a BLACKLIST pattern → FIX.
      4) Otherwise PASS.
    """
    if not commands:
        return {
            "verdict": "FIX",
            "explanation": "No commands provided; nothing to execute safely."
        }

    for c in commands:
        ft = _first_token(c)
        if ft not in SAFE_BINARIES:
            return {
                "verdict": "FIX",
                "explanation": f"Unsafe binary: '{ft}' in command: {c}"
            }

        hit = _find_blacklist_hit(c)
        if hit:
            return {
                "verdict": "FIX",
                "explanation": f"Blacklisted pattern matched ({hit}) in command: {c}"
            }

    return {
        "verdict": "PASS",
        "explanation": "All commands are on the allowlist and contain no blacklisted patterns."
    }
