from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OfficialBaselineSpec:
    key: str
    display_name: str
    repo_url: str
    local_dir: str
    paper: str
    license_name: Optional[str] = None
    supported_types: tuple[str, ...] = ("node",)
    native_adapter: bool = False
    notes: str = ""

    def as_dict(self, root: str | Path) -> dict[str, object]:
        path = Path(root) / "external" / "official_baselines" / self.local_dir
        return {
            "key": self.key,
            "display_name": self.display_name,
            "repo_url": self.repo_url,
            "local_path": str(path),
            "installed": path.exists(),
            "paper": self.paper,
            "license": self.license_name,
            "supported_types": list(self.supported_types),
            "native_adapter": self.native_adapter,
            "notes": self.notes,
        }


OFFICIAL_BASELINE_SPECS: dict[str, OfficialBaselineSpec] = {
    "grapheraser": OfficialBaselineSpec(
        key="grapheraser",
        display_name="GraphEraser / Graph-Unlearning",
        repo_url="https://github.com/MinChen00/Graph-Unlearning.git",
        local_dir="Graph-Unlearning",
        paper="Graph Unlearning, ACM CCS 2022",
        supported_types=("node",),
        native_adapter=True,
        notes="Native BEKM/BLPA adapter is implemented in src/baselines/grapheraser.py.",
    ),
    "gnndelete": OfficialBaselineSpec(
        key="gnndelete",
        display_name="GNNDelete",
        repo_url="https://github.com/mims-harvard/GNNDelete.git",
        local_dir="GNNDelete",
        paper="GNNDelete, ICLR 2023",
        license_name="MIT",
        supported_types=("node", "edge", "feature"),
        native_adapter=False,
        notes="Official repository uses its own preprocessing and deletion operators; a native adapter is not implemented yet.",
    ),
    "gif": OfficialBaselineSpec(
        key="gif",
        display_name="GIF / GIF-torch",
        repo_url="https://github.com/wujcan/GIF-torch.git",
        local_dir="GIF-torch",
        paper="Graph-oriented Influence Function, WWW 2023",
        supported_types=("edge",),
        native_adapter=True,
        notes="Native edge-unlearning adapter is implemented in src/baselines/gif.py using this project's forget-set protocol.",
    ),
}


def get_official_spec(key: str) -> OfficialBaselineSpec:
    normalized = key.lower()
    if normalized in {"grapheraser-bekm", "grapheraser-blpa"}:
        normalized = "grapheraser"
    if normalized not in OFFICIAL_BASELINE_SPECS:
        supported = ", ".join(sorted(OFFICIAL_BASELINE_SPECS))
        raise KeyError(f"Unknown official baseline {key!r}. Supported: {supported}")
    return OFFICIAL_BASELINE_SPECS[normalized]


def official_specs_as_dict(root: str | Path) -> dict[str, dict[str, object]]:
    return {key: spec.as_dict(root) for key, spec in OFFICIAL_BASELINE_SPECS.items()}
