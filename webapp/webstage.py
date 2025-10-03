# webapp/webstage.py
from pathlib import Path
import shutil, subprocess, sys

APP = Path(__file__).resolve().parent  # .../Splendor-AI/webapp
ROOT = APP.parent  # repo root
SRC  = ROOT / "Splendor"  # top-level package
DST  = APP / "Splendor"  # vendored package inside webapp

WEIGHTS_SRC = SRC / "RL" / "trained_agents" / "inference_model.npz"
WEIGHTS_DST = DST / "RL" / "trained_agents" / "inference_model.npz"

def ignore(dirpath, names):
    junk = {
        "__pycache__", ".mypy_cache", ".pytest_cache", ".git", ".DS_Store",
        ".txt"
    }
    return {n for n in names if n in junk or n.endswith((".pyc", ".pyo"))}

if not SRC.exists():
    sys.exit(f"[error] Missing source package: {SRC}")

# Clean & copy
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=ignore)
print("[stage] Copied Splendor ->", DST)

# Weights
if WEIGHTS_SRC.exists():
    WEIGHTS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEIGHTS_SRC, WEIGHTS_DST)
    print("[stage] Copied weights to", WEIGHTS_DST)
else:
    print("[stage] NOTE: weights not found, skipping:", WEIGHTS_SRC)

# Build
cmd = ["pygbag", "--PYBUILD", "3.12", str(APP / "main.py")]
print("[build]", " ".join(cmd))
subprocess.check_call(cmd)
print("[done] Built:", APP / "build" / "web")
