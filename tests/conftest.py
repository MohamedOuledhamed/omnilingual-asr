import sys
from pathlib import Path

# Ensure the project src directory is on the import path for tests
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
