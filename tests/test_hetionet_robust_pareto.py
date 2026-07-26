import json
from types import SimpleNamespace

import pytest

from experiments import sweep_hasi_params as sweep


def _assert_all_components_active(configs):
    assert len(configs) == len(
        {tuple(config[key] for key in sweep.CONFIG_KEYS) for config in configs}
    )
    for config in configs:
        assert config["anchor_lambda1"] > 0
        assert config["anchor_lambda2"] > 0
        assert config["forget_weight"] > 0
        assert config["edge_forget_loss_mode"] != "none"
        assert config["finetune_lr"] > 0
        assert config["finetune_epochs"] > 0
        assert config["inpainting_repair_ratio"] > 0
        assert config["inpainting_max_added_edges"] > 0


@pytest.mark.parametrize(
    "grid_name",
    [
        "edge_hetionet_robust_pareto_refine",
        "node_hetionet_robust_pareto_refine",
    ],
)
def test_hetionet_robust_grids_have_14_active_component_configs(grid_name):
    configs = sweep._grid(grid_name)

    assert len(configs) == 14
    _assert_all_components_active(configs)


def _run_row(config_id, ratio, seed, *, is_default, privacy_gap, val_drop, runtime):
    row = {
        "config_id": config_id,
        "is_default_reference": is_default,
        "dataset": "hetionet-full-nosource",
        "unlearning_type": "node",
        "ratio": ratio,
        "seed": seed,
        "status": "ok",
        "privacy_applicable": True,
        "overall_mia_auc": 0.55,
        "privacy_gap": privacy_gap,
        "privacy_score": 1.0 - privacy_gap,
        "val_accuracy_drop": val_drop,
        "val_f1_macro_drop": val_drop,
        "test_accuracy_drop": val_drop,
        "test_f1_macro_drop": val_drop,
        "degree_kl_abs": 0.0,
        "clustering_change_abs": 0.0,
        "component_change_abs": 0.0,
        "exact_retrain_js_mean": 0.0,
        "exact_retrain_tv_mean": 0.0,
        "exact_retrain_disagreement_rate": 0.0,
        "unlearn_time_seconds": runtime,
    }
    row.update(sweep.DEFAULT_CONFIG)
    return row


def _constraint_args():
    return SimpleNamespace(
        paired_hard_constraints=True,
        accuracy_drop_limit=0.10,
        f1_macro_drop_limit=0.10,
        mia_auc_limit=0.90,
        require_privacy_improvement_vs_default=True,
        privacy_gap_improvement_margin=0.005,
        val_accuracy_drop_delta_limit_vs_default=0.0,
        val_f1_macro_drop_delta_limit_vs_default=0.0,
        test_accuracy_drop_delta_limit_vs_default=0.0,
        test_f1_macro_drop_delta_limit_vs_default=0.0,
        degree_kl_abs_delta_limit_vs_default=0.0,
        clustering_change_abs_delta_limit_vs_default=0.0,
        component_change_abs_delta_limit_vs_default=0.0,
        exact_retrain_js_delta_limit_vs_default=None,
        exact_retrain_tv_delta_limit_vs_default=None,
        exact_retrain_disagreement_delta_limit_vs_default=None,
        unlearn_time_delta_limit_vs_default=None,
        runtime_ratio_mean_limit_vs_default=1.05,
    )


def test_paired_constraints_reject_a_single_seed_regression_hidden_by_mean():
    rows = [
        _run_row("default", 0.05, 42, is_default=True, privacy_gap=0.10, val_drop=0.02, runtime=10.0),
        _run_row("default", 0.05, 123, is_default=True, privacy_gap=0.10, val_drop=0.02, runtime=10.0),
        _run_row("candidate", 0.05, 42, is_default=False, privacy_gap=0.08, val_drop=0.01, runtime=10.0),
        _run_row("candidate", 0.05, 123, is_default=False, privacy_gap=0.105, val_drop=0.01, runtime=10.0),
    ]

    scored = sweep._score_run_rows(rows, SimpleNamespace(runtime_multiplier=1.05))
    aggregates = sweep._aggregate_configs(scored)
    sweep._add_default_relative_metrics(aggregates)
    candidate = next(row for row in aggregates if row["config_id"] == "candidate")

    assert candidate["privacy_gap_delta_vs_default"] < -0.005
    assert candidate["privacy_gap_paired_delta_vs_default_max"] == pytest.approx(0.005)
    assert sweep._passes_config_constraints(candidate, _constraint_args()) is False


def test_paired_constraints_accept_consistent_privacy_gain_without_other_regressions():
    rows = [
        _run_row("default", 0.05, 42, is_default=True, privacy_gap=0.10, val_drop=0.02, runtime=10.0),
        _run_row("default", 0.10, 123, is_default=True, privacy_gap=0.12, val_drop=0.03, runtime=10.0),
        _run_row("candidate", 0.05, 42, is_default=False, privacy_gap=0.09, val_drop=0.02, runtime=10.2),
        _run_row("candidate", 0.10, 123, is_default=False, privacy_gap=0.11, val_drop=0.03, runtime=10.2),
    ]

    scored = sweep._score_run_rows(rows, SimpleNamespace(runtime_multiplier=1.05))
    aggregates = sweep._aggregate_configs(scored)
    sweep._add_default_relative_metrics(aggregates)
    candidate = next(row for row in aggregates if row["config_id"] == "candidate")

    assert candidate["privacy_gap_paired_delta_vs_default_max"] == pytest.approx(-0.01)
    assert candidate["runtime_ratio_vs_default_max"] == pytest.approx(1.02)
    assert sweep._passes_config_constraints(candidate, _constraint_args()) is True


def test_reference_config_file_replaces_the_builtin_default(tmp_path):
    reference = {
        **sweep.DEFAULT_CONFIG,
        "anchor_lambda1": 0.1,
        "anchor_lambda2": 0.02,
        "node_forget_loss_mode": "uniform",
        "forget_weight": 0.8,
        "finetune_lr": 0.003,
        "finetune_epochs": 120,
        "inpainting_repair_ratio": 0.05,
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    loaded = sweep._reference_config_from_file(str(path))

    assert loaded == reference
    assert loaded is not sweep.DEFAULT_CONFIG
