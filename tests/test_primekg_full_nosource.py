from __future__ import annotations

import copy
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import primekg


class _FakeData:
    def __init__(self):
        self.x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        self.y = torch.tensor([0, 1, 0])
        self.edge_index = torch.tensor([[0, 1], [1, 2]])
        self.num_nodes = 3
        self.primekg_node_source = ["a", "b", "a"]

    def clone(self):
        return copy.deepcopy(self)


def test_full_nosource_preserves_graph_and_removes_only_source_suffix(tmp_path, monkeypatch):
    source_data = _FakeData()
    source_metadata = {
        "dataset": "primekg",
        "dataset_variant": "primekg_homo",
        "processed_data": "source/data.pt",
        "relation_mapping": {"r0": 0, "r1": 1},
        "source_mapping": {"a": 0, "b": 1},
        "num_nodes": 3,
        "num_edges_processed": 2,
        "num_features": 7,
        "num_classes": 2,
    }
    monkeypatch.setattr(
        primekg,
        "load_primekg_homo",
        lambda *args, **kwargs: (source_data, source_metadata),
    )

    data, metadata = primekg.load_primekg_full_nosource(tmp_path)

    assert data.x.shape == (3, 3)
    assert torch.equal(data.x, source_data.x[:, :3])
    assert torch.equal(data.y, source_data.y)
    assert torch.equal(data.edge_index, source_data.edge_index)
    assert not hasattr(data, "primekg_node_source")
    assert metadata["dataset"] == "primekg-full-nosource"
    assert metadata["source_feature_removed"] is True
    assert metadata["removed_source_feature_count"] == 2
    assert metadata["relation_type_features_removed"] is True
    assert metadata["removed_relation_feature_count"] == 2
    assert metadata["feature_protocol"] == "structure_only_v1"
    assert metadata["num_features"] == 3
    assert "node_source_one_hot" not in metadata["feature_schema"]
    assert "normalized_relation_type_counts" not in metadata["feature_schema"]

    cached_data, cached_metadata = primekg.load_primekg_full_nosource(tmp_path)
    assert torch.equal(cached_data.x, data.x)
    assert cached_metadata == metadata
