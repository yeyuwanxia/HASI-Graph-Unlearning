from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "opengu_adapted_baselines" / "src", REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from opengu_adapted_baselines import OpenGUAdaptedGIF


def test_opengu_gif_uses_bundled_default_parameters() -> None:
    config = OpenGUAdaptedGIF().config

    assert config.iteration == 100
    assert config.scale == 1_000_000_000.0
    assert config.damp == 0.0
    assert config.hops == 2
