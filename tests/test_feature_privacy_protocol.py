from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import load_forget_set, save_forget_set
from evaluation.metrics import build_experiment_metrics


def test_global_feature_request_disables_node_mia_and_reports_compliance():
    before = SimpleNamespace(
        num_nodes=4,
        x=torch.tensor([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )
    after = SimpleNamespace(**before.__dict__)
    after.x = before.x.clone()
    after.x[:, 0] = 0
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
    embeddings = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    metrics = build_experiment_metrics(
        method="toy",
        dataset="toy",
        unlearning_type="feature",
        data_before=before,
        data_after=after,
        logits_before=logits,
        logits_after=logits,
        embeddings_before=embeddings,
        embeddings_after=embeddings,
        forget_targets=[0],
    )

    privacy = metrics["privacy"]
    assert privacy["status"] == "not_applicable_global_feature_dimension_request"
    assert privacy["applicable"] is False
    assert privacy["num_members"] == 0
    assert privacy["num_non_members"] == 0
    assert privacy["strong_auc"] is None
    assert metrics["feature_compliance"]["request_applied"] is True
    assert metrics["feature_compliance"]["forgotten_feature_residual_l1_ratio"] == 0.0


def test_edge_protocol_freezes_directed_sampling_and_undirected_deletion(tmp_path):
    path = tmp_path / "edge.json"
    save_forget_set(
        path,
        dataset="toy",
        unlearning_type="edge",
        ratio=0.05,
        seed=42,
        selection="random_all",
        targets=[(0, 1)],
        protocol_metadata={"selection_scope": "train_subgraph_edges"},
    )

    protocol = load_forget_set(path).as_dict()["protocol"]
    assert protocol["sampling_unit"] == "directed_edge_index_entry"
    assert protocol["ratio_denominator"] == "directed_candidate_edge_index_entries"
    assert protocol["deletion_operator"] == "undirected_closure"
