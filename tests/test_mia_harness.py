"""MIA harness capability-gate tests.

Runs with ONLY numpy (no torch / networkx). We load privacy_evaluator.py directly
by file path, because importing it via the `evaluation` package would trigger
src/evaluation/__init__.py -> metrics.py -> `import networkx` / `import torch`.

Run:  python tests/test_mia_harness.py      (or: python -m pytest tests/test_mia_harness.py -v)
"""
import importlib.util
import pathlib
import sys

import numpy as np

_p = pathlib.Path(__file__).resolve().parents[1] / "src" / "evaluation" / "mia" / "privacy_evaluator.py"
_spec = importlib.util.spec_from_file_location("privacy_evaluator_direct", _p)
_pe = importlib.util.module_from_spec(_spec)
sys.modules["privacy_evaluator_direct"] = _pe  # required for @dataclass resolution
_spec.loader.exec_module(_pe)
PrivacyEvaluator = _pe.PrivacyEvaluator


def _run(before, after, members, non_members, **kw):
    return PrivacyEvaluator().evaluate_from_logits(before, after, members, non_members, **kw)


def test_random_model_is_near_null():
    # No systematic before/after difference -> attacker should NOT be significant.
    rng = np.random.default_rng(0)
    n, c = 400, 3
    before = rng.normal(size=(n, c))
    after = before + rng.normal(scale=0.01, size=(n, c))
    r = _run(before, after, members=list(range(50)), non_members=list(range(200, 250)))
    assert r.strong_auc_pvalue > 0.05, r.strong_auc_pvalue


def test_memorized_members_are_detectable():
    # Large random perturbation on members only -> attacker should detect it (CAPABILITY GATE).
    rng = np.random.default_rng(1)
    n, c = 400, 3
    before = rng.normal(size=(n, c))
    after = before.copy()
    members = list(range(50))
    after[members] += rng.normal(scale=5.0, size=(len(members), c))
    r = _run(before, after, members=members, non_members=list(range(200, 250)))
    # Judge on the permutation p-value + being above the null mean, NOT a hard AUC threshold.
    assert r.strong_auc_pvalue < 0.05, r.strong_auc_pvalue
    assert r.strong_auc > r.strong_auc_null_mean, (r.strong_auc, r.strong_auc_null_mean)


def test_label_shuffle_kills_signal():
    # Point "members" at unchanged nodes -> signal should vanish (negative control).
    rng = np.random.default_rng(2)
    n, c = 400, 3
    before = rng.normal(size=(n, c))
    after = before.copy()
    idx = list(range(50))
    after[idx] += rng.normal(scale=5.0, size=(50, c))
    r = _run(before, after, members=list(range(200, 250)), non_members=list(range(250, 300)))
    assert r.strong_auc_pvalue > 0.05, r.strong_auc_pvalue


def test_medium_attack_evaluates_only_held_out_rows():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 4))
    members = np.arange(0, 10)
    non_members = np.arange(20, 30)

    scores, eval_idx, uses_shadow, train_size = _pe.MediumAttacker().scores(
        features, members, non_members
    )

    y_true = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
    train_idx = _pe._stratified_half_indices(y_true)
    assert uses_shadow is False
    assert train_size == len(train_idx)
    assert len(eval_idx) + train_size == len(y_true)
    assert not set(eval_idx).intersection(train_idx)
    assert scores.shape == (len(eval_idx),)


def _pairwise_auc_reference(y_true, scores):
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    for positive in positives:
        wins += np.sum(positive > negatives)
        wins += 0.5 * np.sum(positive == negatives)
    return wins / float(len(positives) * len(negatives))


def test_rank_auc_matches_pairwise_reference_with_ties_and_permutations():
    rng = np.random.default_rng(23)
    scores = rng.integers(-3, 4, size=240).astype(float)
    labels = np.concatenate([np.ones(120), np.zeros(120)])
    rng.shuffle(labels)

    expected = _pairwise_auc_reference(labels, scores)
    assert np.isclose(_pe._auc(labels, scores), expected)

    ranks = _pe._average_ranks(scores)
    for _ in range(20):
        permuted = rng.permutation(labels)
        expected = _pairwise_auc_reference(permuted, scores)
        assert np.isclose(_pe._auc_from_ranks(permuted, ranks), expected)
        assert np.isclose(
            _pe._attack_auc_from_ranks(permuted, ranks),
            max(expected, 1.0 - expected),
        )

if __name__ == "__main__":
    test_random_model_is_near_null()
    test_memorized_members_are_detectable()
    test_label_shuffle_kills_signal()
    print("all MIA harness gate tests passed")
