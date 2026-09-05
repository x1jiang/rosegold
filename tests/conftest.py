import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Route batch/audit/criteria writes to a throwaway directory unless the caller
# deliberately pointed ROSEGOLD_OUTPUT_DIR somewhere. Keeps `pytest` from
# polluting outputs/ in the working tree.
if not os.getenv("ROSEGOLD_OUTPUT_DIR"):
    _TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="rosegold-test-outputs-")
    os.environ["ROSEGOLD_OUTPUT_DIR"] = _TEST_OUTPUT_DIR
