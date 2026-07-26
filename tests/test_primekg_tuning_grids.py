from experiments import sweep_hasi_params as sweep


def _assert_complete_hasi_grid(configs):
    assert len(configs) == len({tuple(config[key] for key in sweep.CONFIG_KEYS) for config in configs})
    for config in configs:
        assert config["anchor_lambda1"] > 0
        assert config["anchor_lambda2"] > 0
        assert config["forget_weight"] > 0
        assert config["finetune_epochs"] > 0
        assert config["finetune_lr"] > 0
        assert config["inpainting_repair_ratio"] > 0
        assert config["inpainting_max_added_edges"] > 0


def test_primekg_edge_grid_is_diverse_and_keeps_components_active():
    configs = sweep._grid("edge_primekg_privacy_guarded_refine")

    assert len(configs) == 14
    assert {config["edge_forget_loss_mode"] for config in configs} == {"original_kl", "uniform"}
    assert len({config["forget_weight"] for config in configs}) >= 4
    assert len({(config["anchor_lambda1"], config["anchor_lambda2"]) for config in configs}) >= 4
    _assert_complete_hasi_grid(configs)


def test_primekg_feature_grid_is_default_centered_and_keeps_components_active():
    configs = sweep._grid("feature_primekg_utility_guarded_refine")

    assert len(configs) == 15
    assert any(config["anchor_lambda1"] == 2.0 and config["anchor_lambda2"] == 0.5 for config in configs)
    assert any(config["finetune_lr"] == 0.01 for config in configs)
    assert {config["edge_forget_loss_mode"] for config in configs} == {"original_kl", "uniform"}
    _assert_complete_hasi_grid(configs)


def test_primekg_node_grid_controls_the_actual_node_loss_mode():
    configs = sweep._grid("node_primekg_privacy_guarded_refine")

    assert len(configs) == 14
    assert {config["node_forget_loss_mode"] for config in configs} == {
        "original_kl",
        "uniform",
    }
    _assert_complete_hasi_grid(configs)
