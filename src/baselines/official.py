from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from .baselines import BaselineRunResult
from .official_sources import OfficialBaselineSpec


class OfficialBaselineUnavailable(RuntimeError):
    pass


class ExternalOfficialBaseline:
    """Guardrail adapter for official baselines that are not natively wired.

    The project should not silently report fine-tuning surrogates as official
    baselines. Until a repository-specific native adapter converts our PyG data,
    forget-set protocol, and metrics into the official code's format, these
    entries fail loudly with installation and implementation guidance.
    """

    def __init__(self, spec: OfficialBaselineSpec, root: str | Path):
        self.spec = spec
        self.root = Path(root)
        self.name = spec.key

    def run_node_unlearning(self, data, forget_nodes: Iterable[int], **kwargs: Any) -> BaselineRunResult:
        self._raise("node")

    def run_edge_unlearning(self, data, forget_edges: Iterable[tuple[int, int]], **kwargs: Any) -> BaselineRunResult:
        self._raise("edge")

    def run_feature_unlearning(self, data, forget_features: Iterable[int], **kwargs: Any) -> BaselineRunResult:
        self._raise("feature")

    def _raise(self, unlearning_type: str) -> None:
        local_path = self.root / "external" / "official_baselines" / self.spec.local_dir
        installed = local_path.exists()
        supported = ", ".join(self.spec.supported_types)
        raise OfficialBaselineUnavailable(
            "\n".join(
                [
                    f"{self.spec.display_name} is registered as an official baseline, but no native adapter is implemented yet.",
                    f"Requested unlearning_type={unlearning_type!r}; official supported_types=[{supported}].",
                    f"Official repository: {self.spec.repo_url}",
                    f"Local path: {local_path} ({'installed' if installed else 'not installed'})",
                    "Use experiments/install_official_baselines.py to fetch the source, then implement a repository-specific adapter",
                    "that consumes this project's forget-set files and returns BaselineRunResult.",
                    f"For non-official sanity checks, use --baseline {self.spec.key}-surrogate if available.",
                ]
            )
        )

    def as_dict(self) -> dict[str, Any]:
        local_path = self.root / "external" / "official_baselines" / self.spec.local_dir
        return {
            "method": self.name,
            "official": True,
            "native_adapter": False,
            "repo_url": self.spec.repo_url,
            "local_path": str(local_path),
            "installed": local_path.exists(),
        }
