from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.generate_forget_sets import _select_targets, _select_unique_undirected_edges


def _data():
    return SimpleNamespace(
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 0],
                [1, 0, 2, 1, 3, 2, 3],
            ],
            dtype=torch.long,
        ),
        train_mask=torch.tensor([True, True, True, False]),
    )


def test_unique_undirected_selection_is_canonical_and_deterministic():
    first = _select_unique_undirected_edges(
        _data(), 0.5, seed=7, train_subgraph=False
    )
    second = _select_unique_undirected_edges(
        _data(), 0.5, seed=7, train_subgraph=False
    )

    targets, candidate_count, directed_candidate_count = first
    assert first == second
    assert candidate_count == 4
    assert directed_candidate_count == 7
    assert len(targets) == 2
    assert all(source <= target for source, target in targets)
    assert len(targets) == len(set(targets))


def test_unique_undirected_protocol_uses_unique_train_edge_denominator():
    args = SimpleNamespace(
        unlearning_type="edge",
        selection="random_all",
        edge_scope="train_subgraph",
        edge_sampling_unit="unique_undirected",
        forget_ratio=0.5,
        seed=11,
    )

    targets, protocol = _select_targets(_data(), args)

    assert len(targets) == 1
    assert protocol["candidate_count"] == 2
    assert protocol["directed_candidate_entry_count"] == 4
    assert protocol["sampling_unit"] == "unique_undirected_edge"
    assert protocol["ratio_denominator"] == "unique_candidate_undirected_edges"
    assert protocol["deletion_operator"] == "undirected_closure"
