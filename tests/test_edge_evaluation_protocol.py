from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import edge_forgetting_metrics, exact_retrain_alignment_metrics, structural_metrics
from evaluation.retrain_reference import load_exact_retrain_reference, save_exact_retrain_reference


def test_edge_forgetting_scores_request_and_retrain_gap() -> None:
    graph_before = nx.Graph([(0, 1), (1, 2)])
    graph_before.add_nodes_from(range(3))
    graph_after = nx.Graph([(1, 2)])
    graph_after.add_nodes_from(range(3))
    embeddings_before = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    embeddings_after = np.array([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    result = edge_forgetting_metrics(
        graph_before,
        graph_after,
        embeddings_before,
        embeddings_after,
        [(1, 0), (0, 1)],
        exact_retrain_embeddings=embeddings_after,
        seed=7,
    )

    assert result["status"] == "ok"
    assert result["request_applied"] is True
    assert result["forgotten_edge_count"] == 1
    assert result["forgotten_score_drop_mean"] == pytest.approx(1.0)
    assert result["retained_control_score_drop_mean"] == pytest.approx(0.0)
    assert result["targeted_drop_vs_control"] == pytest.approx(1.0)
    assert result["forgotten_unlearned_to_retrain_abs_gap_mean"] == pytest.approx(0.0)


def test_exact_retrain_alignment_rewards_matching_outputs() -> None:
    logits_before = np.array([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    logits_after = np.array([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]])
    data = SimpleNamespace(test_mask=np.array([True, True, False]))

    result = exact_retrain_alignment_metrics(
        logits_before,
        logits_after,
        logits_after.copy(),
        data,
        reference={
            "schema_version": "exact_retrain_reference_v1",
            "artifact_path": "reference.pt",
            "forget_set_sha256": "abc",
        },
    )

    assert result["status"] == "ok"
    assert result["unlearned_to_retrain_js_mean"] == pytest.approx(0.0)
    assert result["unlearned_to_retrain_tv_mean"] == pytest.approx(0.0)
    assert result["improvement_over_original_js"] > 0
    assert result["prediction_disagreement_rate"] == pytest.approx(0.0)
    assert result["reference_path"] == "reference.pt"



def test_parallel_clustering_matches_networkx_exactly(monkeypatch) -> None:
    graph_before = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    graph_before.add_node(5)
    graph_after = graph_before.copy()
    graph_after.remove_edge(0, 1)
    monkeypatch.setenv("STRUCTURAL_METRICS_WORKERS", "2")
    monkeypatch.setenv("STRUCTURAL_METRICS_PARALLEL_MIN_EDGES", "0")

    result = structural_metrics(graph_before, graph_after)

    assert result["clustering_backend"] == "networkx_fork_pool_degree_balanced"
    assert result["clustering_workers"] == 2
    assert result["clustering_coefficient_before"] == pytest.approx(
        nx.average_clustering(graph_before), abs=1e-15
    )
    assert result["clustering_coefficient_after"] == pytest.approx(
        nx.average_clustering(graph_after), abs=1e-15
    )
def test_exact_retrain_reference_round_trip_and_protocol_binding(tmp_path: Path) -> None:
    forget_set = tmp_path / "forget.json"
    forget_set.write_text('{"targets": [[0, 1]]}\n', encoding="utf-8")
    artifact = tmp_path / "reference.pt"
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    metadata = save_exact_retrain_reference(
        artifact,
        logits=logits,
        embeddings=embeddings,
        dataset="pubmed",
        unlearning_type="edge",
        forget_set_path=forget_set,
        base_artifact_path="results/shared_base/pubmed/seed42",
        seed=42,
        model_config={"type": "GCN"},
        training={"epochs_ran": 1},
    )
    loaded = load_exact_retrain_reference(
        artifact,
        dataset="pubmed",
        unlearning_type="edge",
        forget_set_path=forget_set,
        base_artifact_path="results/shared_base/pubmed/seed42",
    )

    assert metadata["schema_version"] == "exact_retrain_reference_v1"
    assert torch.equal(loaded["logits"], logits)
    assert torch.equal(loaded["embeddings"], embeddings)
    assert artifact.with_suffix(".pt.json").is_file()

    forget_set.write_text('{"targets": [[1, 2]]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="forget_set_sha256"):
        load_exact_retrain_reference(
            artifact,
            dataset="pubmed",
            unlearning_type="edge",
            forget_set_path=forget_set,
            base_artifact_path="results/shared_base/pubmed/seed42",
        )
