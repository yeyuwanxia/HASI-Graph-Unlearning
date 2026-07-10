from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ForgetSet:
    dataset: Optional[str]
    unlearning_type: str
    ratio: Optional[float]
    seed: Optional[int]
    selection: Optional[str]
    targets: list[Any]
    path: Optional[Path] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "unlearning_type": self.unlearning_type,
            "ratio": self.ratio,
            "seed": self.seed,
            "selection": self.selection,
            "targets": self.targets,
        }


def save_forget_set(
    path: str | Path,
    *,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    seed: int,
    selection: str,
    targets: Iterable[Any],
    protocol_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = ForgetSet(
        dataset=_normalize_dataset_name(dataset),
        unlearning_type=_normalize_type(unlearning_type),
        ratio=float(ratio),
        seed=int(seed),
        selection=str(selection),
        targets=_normalize_targets(targets, unlearning_type),
    )
    payload = spec.as_dict()
    if protocol_metadata:
        payload["protocol"] = protocol_metadata
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_forget_set(
    path: str | Path,
    *,
    expected_type: Optional[str] = None,
    expected_dataset: Optional[str] = None,
) -> ForgetSet:
    input_path = Path(path)
    content = input_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Forget set file is empty: {input_path}")

    if input_path.suffix.lower() == ".json" or content.startswith("{"):
        payload = json.loads(content)
        unlearning_type = _normalize_type(payload.get("unlearning_type") or expected_type or "")
        if not unlearning_type:
            raise ValueError("JSON forget set needs 'unlearning_type' or an expected_type.")
        dataset = payload.get("dataset")
        spec = ForgetSet(
            dataset=_normalize_dataset_name(dataset) if dataset else None,
            unlearning_type=unlearning_type,
            ratio=_optional_float(payload.get("ratio")),
            seed=_optional_int(payload.get("seed")),
            selection=payload.get("selection"),
            targets=_normalize_targets(payload.get("targets", []), unlearning_type),
            path=input_path,
        )
    else:
        unlearning_type = _normalize_type(expected_type or "")
        if not unlearning_type:
            raise ValueError("Text forget set files require expected_type.")
        spec = ForgetSet(
            dataset=None,
            unlearning_type=unlearning_type,
            ratio=None,
            seed=None,
            selection="text_file",
            targets=_parse_text_targets(content, unlearning_type),
            path=input_path,
        )

    if expected_type and spec.unlearning_type != _normalize_type(expected_type):
        raise ValueError(
            f"Forget set type mismatch: file has {spec.unlearning_type!r}, "
            f"expected {_normalize_type(expected_type)!r}."
        )
    if expected_dataset and spec.dataset:
        expected = _normalize_dataset_name(expected_dataset)
        if spec.dataset != expected:
            raise ValueError(
                f"Forget set dataset mismatch: file has {spec.dataset!r}, expected {expected!r}."
            )
    return spec


def parse_forget_targets(value: str, unlearning_type: str) -> list[Any]:
    if not value:
        return []
    unlearning_type = _normalize_type(unlearning_type)
    if unlearning_type == "edge":
        return _parse_edges(value.replace("\n", ",").split(","))
    return _parse_ints(value.replace("\n", ",").split(","))


def default_forget_set_path(
    root: str | Path,
    *,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    seed: int,
    selection: str,
) -> Path:
    ratio_label = str(ratio).replace(".", "p")
    name = f"{_normalize_dataset_name(dataset)}_{_normalize_type(unlearning_type)}_r{ratio_label}_{selection}_seed{seed}.json"
    return Path(root) / "experiments" / "forget_sets" / name


def _normalize_targets(targets: Iterable[Any], unlearning_type: str) -> list[Any]:
    unlearning_type = _normalize_type(unlearning_type)
    if unlearning_type == "edge":
        normalized = []
        for edge in targets:
            if isinstance(edge, str):
                normalized.extend(_parse_edges([edge]))
                continue
            source, target = edge
            normalized.append([int(source), int(target)])
        return normalized
    return [int(target) for target in targets]


def _parse_text_targets(content: str, unlearning_type: str) -> list[Any]:
    rows = [line.strip() for line in content.replace(";", "\n").splitlines() if line.strip()]
    if _normalize_type(unlearning_type) == "edge":
        return _parse_edges(rows)
    return _parse_ints(rows)


def _parse_edges(values: Iterable[str]) -> list[list[int]]:
    edges: list[list[int]] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        source, target = item.replace(":", "-").replace(",", "-").split("-", maxsplit=1)
        edges.append([int(source), int(target)])
    return edges


def _parse_ints(values: Iterable[str]) -> list[int]:
    return [int(str(value).strip()) for value in values if str(value).strip()]


def _normalize_type(unlearning_type: str) -> str:
    value = str(unlearning_type).lower().strip()
    if value not in {"node", "edge", "feature"}:
        raise ValueError(f"Unsupported unlearning_type {unlearning_type!r}.")
    return value


def _normalize_dataset_name(dataset: str) -> str:
    return str(dataset).lower().replace("_", "-")


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


def _optional_int(value) -> Optional[int]:
    return None if value is None else int(value)
