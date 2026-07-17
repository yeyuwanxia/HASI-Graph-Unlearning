from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import infer_non_member_nodes


def _toy_data():
    return SimpleNamespace(
        num_nodes=10,
        x=np.zeros((10, 2)),
        train_mask=np.array([True, True, True, True, True, False, False, False, False, False]),
        y=np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1]),
    )


def test_train_members_get_train_non_members():
    data = _toy_data()
    members = [0, 2]
    sampled = infer_non_member_nodes(data, members, 3, seed=7, labels=data.y)
    assert len(sampled) == 3
    assert not set(sampled).intersection(members)
    assert all(bool(data.train_mask[node]) for node in sampled)


def test_non_train_members_keep_all_node_control_pool():
    data = _toy_data()
    members = [7]
    sampled = infer_non_member_nodes(data, members, 5, seed=7, labels=data.y)
    assert len(sampled) == 5
    assert 7 not in sampled


if __name__ == "__main__":
    test_train_members_get_train_non_members()
    test_non_train_members_keep_all_node_control_pool()
    print("all MIA non-member sampling tests passed")
