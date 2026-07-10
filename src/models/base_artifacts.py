from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .trainer import GNNTrainer, TrainingResult


MODEL_STATE_FILE = "model_state.pt"
LOGITS_FILE = "logits.pt"
EMBEDDINGS_FILE = "embeddings.pt"
METADATA_FILE = "metadata.json"


def default_base_artifact_dir(root: str | Path, dataset_name: str, seed: int | str) -> Path:
    return Path(root) / dataset_name / f"seed{seed}"


def save_base_artifact(
    artifact_dir: str | Path,
    *,
    trainer: GNNTrainer,
    data,
    dataset_name: str,
    seed: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    training_result: TrainingResult | dict[str, Any],
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    logits, embeddings = trainer.predict_with_embeddings(data)
    result_dict = training_result.as_dict() if hasattr(training_result, "as_dict") else dict(training_result)
    metadata = {
        "dataset": dataset_name,
        "seed": int(seed),
        "model": model_config,
        "training": training_config,
        "base_training": result_dict,
        "num_nodes": int(data.num_nodes),
        "num_features": int(data.num_features),
        "num_classes": int(getattr(data, "y").max().item() + 1) if hasattr(data, "y") else None,
        "mask_counts": _mask_counts(data),
        "files": {
            "model_state": MODEL_STATE_FILE,
            "logits": LOGITS_FILE,
            "embeddings": EMBEDDINGS_FILE,
        },
    }

    torch.save(trainer.model.state_dict(), artifact_path / MODEL_STATE_FILE)
    torch.save(logits.cpu(), artifact_path / LOGITS_FILE)
    torch.save(embeddings.cpu(), artifact_path / EMBEDDINGS_FILE)
    (artifact_path / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_base_artifact(
    artifact_dir: str | Path,
    *,
    trainer: GNNTrainer,
    dataset_name: str | None = None,
    seed: int | None = None,
    model_config: dict[str, Any] | None = None,
    strict: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing base artifact metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_name is not None and metadata.get("dataset") != dataset_name:
        raise ValueError(f"Base artifact dataset mismatch: expected {dataset_name}, got {metadata.get('dataset')}")
    if seed is not None and int(metadata.get("seed")) != int(seed):
        raise ValueError(f"Base artifact seed mismatch: expected {seed}, got {metadata.get('seed')}")
    if model_config is not None:
        _validate_model_config(metadata.get("model", {}), model_config)

    state = torch.load(artifact_path / MODEL_STATE_FILE, map_location=trainer.device)
    trainer.model.load_state_dict(state, strict=strict)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    logits = torch.load(artifact_path / LOGITS_FILE, map_location="cpu")
    embeddings = torch.load(artifact_path / EMBEDDINGS_FILE, map_location="cpu")
    return metadata, logits, embeddings


def _validate_model_config(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = ("type", "hidden_channels", "num_layers", "dropout", "in_channels", "out_channels")
    mismatches = []
    for key in keys:
        if key in saved and key in expected and saved[key] != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, got {saved[key]!r}")
    if mismatches:
        raise ValueError("Base artifact model mismatch: " + "; ".join(mismatches))

def _mask_counts(data) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is None:
            counts[name] = None
            continue
        if mask.dim() > 1:
            mask = mask[:, 0]
        counts[name] = int(mask.detach().cpu().to(dtype=torch.bool).sum().item())
    return counts

