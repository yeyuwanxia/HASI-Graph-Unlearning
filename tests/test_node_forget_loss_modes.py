from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_hasi import _resolve_runtime_config, parse_args
from hasi.unlearner import HASIConfig, HASIUnlearner


class _FitResult:
    def as_dict(self):
        return {"status": "ok"}


class _RecordingTrainer:
    def __init__(self, original_data, original_logits, post_removal_logits, train_logits):
        self.original_data = original_data
        self.original_logits = original_logits
        self.post_removal_logits = post_removal_logits
        self.train_logits = train_logits
        self.loss = None

    def predict_with_embeddings(self, data):
        logits = self.original_logits if data is self.original_data else self.post_removal_logits
        embeddings = torch.zeros((logits.shape[0], 2), dtype=logits.dtype)
        return logits.clone(), embeddings

    def fine_tune(self, data, *, train_mask, epochs, lr, extra_loss_fn):
        logits = self.train_logits.clone().requires_grad_(True)
        embeddings = torch.zeros((logits.shape[0], 2), dtype=logits.dtype)
        self.loss = extra_loss_fn(logits, embeddings)
        self.loss.backward()
        return _FitResult()


def _run_fine_tune(mode: str):
    original_data = SimpleNamespace(num_nodes=2)
    post_removal_data = SimpleNamespace(num_nodes=2)
    original_logits = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    post_removal_logits = torch.tensor([[0.0, 4.0], [0.0, 1.0]])
    train_logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    trainer = _RecordingTrainer(
        original_data,
        original_logits,
        post_removal_logits,
        train_logits,
    )
    unlearner = HASIUnlearner(
        data=original_data,
        graph=nx.Graph([(0, 1)]),
        config=HASIConfig(anchor_mode="none", subgraph_finetune=False),
    )
    result = unlearner._fine_tune(
        trainer,
        post_removal_data,
        forget_nodes=[0],
        dar_anchors=[],
        finetune_epochs=1,
        finetune_lr=0.01,
        forget_weight=1.0,
        forget_loss_mode=mode,
    )
    return trainer.loss.detach(), result, train_logits, original_logits, post_removal_logits


@pytest.mark.parametrize(
    ("mode", "target_source"),
    [
        ("original_kl", "original_graph_logits"),
        ("post_removal_kl", "post_removal_logits"),
        ("uniform", "uniform_distribution"),
        ("none", "none"),
    ],
)
def test_node_forget_loss_modes_use_and_record_the_requested_target(mode, target_source):
    loss, result, train_logits, original_logits, post_removal_logits = _run_fine_tune(mode)
    selected_log_prob = F.log_softmax(train_logits[[0]], dim=-1)
    if mode == "original_kl":
        target = F.softmax(original_logits[[0]], dim=-1)
        expected = F.kl_div(selected_log_prob, target, reduction="batchmean")
    elif mode == "post_removal_kl":
        target = F.softmax(post_removal_logits[[0]], dim=-1)
        expected = F.kl_div(selected_log_prob, target, reduction="batchmean")
    elif mode == "uniform":
        expected = F.kl_div(
            selected_log_prob,
            torch.full_like(selected_log_prob, 0.5),
            reduction="batchmean",
        )
    else:
        expected = torch.tensor(0.0)

    assert torch.allclose(loss, expected)
    assert result["forget_loss_mode"] == mode
    assert result["forget_loss_target"] == target_source


def test_node_unlearning_passes_its_independent_loss_mode(monkeypatch):
    unlearner = HASIUnlearner(
        graph=nx.Graph([(0, 1), (1, 2)]),
        config=HASIConfig(
            anchor_mode="none",
            inpainting_mode="none",
            node_forget_loss_mode="post_removal_kl",
        ),
    )
    monkeypatch.setattr(
        unlearner,
        "plan_node_unlearning",
        lambda nodes: {
            "forget_nodes": list(nodes),
            "primary_forget_nodes": [],
            "affected_region": [1],
            "affected_region_diagnostics": {},
            "dar_contexts": [],
        },
    )
    captured = {}

    def record_fine_tune(*args, **kwargs):
        captured.update(kwargs)
        return {"forget_loss_mode": kwargs["forget_loss_mode"]}

    monkeypatch.setattr(unlearner, "_fine_tune", record_fine_tune)
    result = unlearner.unlearn_nodes([0])

    assert captured["forget_loss_mode"] == "post_removal_kl"
    assert result["training"]["forget_loss_mode"] == "post_removal_kl"


def test_runtime_config_defaults_node_loss_to_uniform(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_hasi.py"])
    config, resolved = _resolve_runtime_config(parse_args(), {})

    assert HASIConfig().node_forget_loss_mode == "uniform"
    assert config.node_forget_loss_mode == "uniform"
    assert resolved["unlearning"]["node_forget_loss_mode"] == "uniform"


def test_runtime_config_accepts_post_removal_node_loss(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_hasi.py", "--node_forget_loss_mode", "post_removal_kl"],
    )
    config, resolved = _resolve_runtime_config(parse_args(), {})

    assert config.node_forget_loss_mode == "post_removal_kl"
    assert resolved["unlearning"]["node_forget_loss_mode"] == "post_removal_kl"
