# docs/webstage.py
# Stage so that GitHub Pages works, which requires "docs" folder
print(f"[debug] running {__file__}")

from pathlib import Path
import shutil, subprocess, sys

APP = Path(__file__).resolve().parent  # .../Splendor-AI/docs
print(APP)
ROOT = APP.parent  # repo root
SRC = ROOT / "Splendor"  # top-level package
DST = APP / "Splendor"  # vendored package inside docs

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
print("[stage] Copied Splendor to", DST)

# Weights
if WEIGHTS_SRC.exists():
    WEIGHTS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEIGHTS_SRC, WEIGHTS_DST)
    print("[stage] Copied weights to", WEIGHTS_DST)
else:
    print("[stage] NOTE: weights not found, skipping", WEIGHTS_SRC)

# Build
cmd = ["pygbag", "--PYBUILD", "3.12", str(APP / "main.py")]
print("[build]", " ".join(cmd))
subprocess.check_call(cmd)
print("[done] Built:", APP / "build" / "web")
