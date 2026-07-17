from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hasi.hub_identification import HubScoreCache, HubScoreConfig, hub_cache_identity


def _identity(data, model):
    return hub_cache_identity(
        dataset="toy",
        training_seed=42,
        data=data,
        model=model,
        hub_config=HubScoreConfig(),
        gradient_enabled=True,
        gradient_passes=1,
        gradient_dropout=False,
    )


def test_hub_score_cache_round_trip_and_invalidation(tmp_path):
    data = SimpleNamespace(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        x=torch.eye(3),
        train_mask=torch.ones(3, dtype=torch.bool),
        y=torch.tensor([0, 1, 0]),
    )
    model = torch.nn.Linear(3, 2)
    cache = HubScoreCache(tmp_path)
    identity = _identity(data, model)

    assert cache.lookup(identity).hit is False
    stored = cache.store(
        identity,
        {0: 0.1, 1: 0.2, 2: 0.3},
        build_seconds=1.25,
        metadata={"producer": "test"},
    )
    assert stored.as_dict()["offline_preprocessing_seconds"] == 1.25
    loaded = cache.lookup(identity)
    assert loaded.hit is True
    assert loaded.scores == {0: 0.1, 1: 0.2, 2: 0.3}
    assert loaded.metadata["producer"] == "test"
    assert loaded.metadata["offline_preprocessing_seconds"] == 1.25
    assert loaded.metadata["created_at"]

    with torch.no_grad():
        model.weight.add_(1.0)
    assert cache.lookup(_identity(data, model)).hit is False

    data.x[0, 0] = 2.0
    assert cache.lookup(_identity(data, model)).hit is False
