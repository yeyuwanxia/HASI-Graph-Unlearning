from __future__ import annotations

import sys
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LOCAL_ROOT.parent
for path in (LOCAL_ROOT / "src", REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opengu_adapted_baselines.runner import main


if __name__ == "__main__":
    main()
