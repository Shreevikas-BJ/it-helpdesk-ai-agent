# app/tools.py
import subprocess, shlex, platform
from .config import WHITELIST_COMMANDS

_WHITELIST = [c.lower() for c in WHITELIST_COMMANDS]

def safe_shell(cmd: str) -> dict:
    cmd = (cmd or "").strip()
    if not cmd:
        return {"ok": False, "stdout": "", "stderr": "empty command"}

    # Parse once for base command
    try:
        parts = shlex.split(cmd)
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": f"parse error: {e}"}
    if not parts:
        return {"ok": False, "stdout": "", "stderr": "empty parts"}

    base = parts[0].lower()
    if base not in _WHITELIST:
        return {"ok": False, "stdout": "", "stderr": f"blocked: {base}"}

    # --- Special-case echo: emulate in Python so it always works ---
    if base == "echo":
        # everything after 'echo' is the message
        msg = cmd[len(parts[0]):].lstrip()
        return {"ok": True, "stdout": (msg + "\n"), "stderr": ""}

    try:
        if platform.system() == "Windows":
            # Use cmd.exe PATH resolution for built-ins and exe lookups
            # We keep whitelist above to avoid injection risks.
            proc = subprocess.run(
                cmd,                      # pass the string
                capture_output=True,
                text=True,
                timeout=8,
                shell=True                # let cmd.exe resolve the program
            )
        else:
            # On Linux/Mac, call directly without shell
            proc = subprocess.run(
                parts,                    # pass the list
                capture_output=True,
                text=True,
                timeout=8,
                shell=False
            )

        return {
            "ok": proc.returncode == 0,
            "stdout": (proc.stdout or "")[:4000],
            "stderr": (proc.stderr or "")[:2000],
        }

    except FileNotFoundError as e:
        return {"ok": False, "stdout": "", "stderr": f"not found: {e}"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)}
