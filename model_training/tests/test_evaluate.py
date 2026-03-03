"""
test_evaluate.py — pytest tests for model_training/src/evaluate.py
"""

from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from evaluate import compute_metrics, baseline_random, baseline_majority


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 0
        assert m["tn"] == 0
        assert m["fp"] == 2
        assert m["fn"] == 2
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_all_predicted_positive(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 2
        assert m["fp"] == 2
        assert m["fn"] == 0
        assert m["tn"] == 0
        assert m["specificity"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_mixed_results(self):
        # TP=3, TN=2, FP=1, FN=1 → accuracy=5/7, precision=3/4, recall=3/4
        y_true = [1, 1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 1, 0, 0, 0, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 3
        assert m["fn"] == 1
        assert m["tn"] == 2
        assert m["fp"] == 1
        assert m["accuracy"] == pytest.approx(5 / 7, abs=1e-5)
        assert m["precision"] == pytest.approx(3 / 4, abs=1e-5)
        assert m["recall"] == pytest.approx(3 / 4, abs=1e-5)

    def test_f1_is_harmonic_mean_of_precision_recall(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        m = compute_metrics(y_true, y_pred)
        expected_f1 = (
            2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
            if (m["precision"] + m["recall"]) > 0
            else 0.0
        )
        assert m["f1"] == pytest.approx(expected_f1, abs=1e-5)

    def test_specificity_formula(self):
        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0, 0, 1, 0, 1, 0]
        m = compute_metrics(y_true, y_pred)
        # TN=3, FP=1 → specificity = 3/4
        assert m["specificity"] == pytest.approx(0.75, abs=1e-5)

    def test_returns_all_keys(self):
        m = compute_metrics([0, 1], [0, 1])
        expected_keys = {"tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "specificity"}
        assert expected_keys.issubset(set(m.keys()))

    def test_counts_are_integers(self):
        m = compute_metrics([0, 1, 0, 1], [0, 1, 1, 0])
        assert isinstance(m["tp"], int)
        assert isinstance(m["tn"], int)
        assert isinstance(m["fp"], int)
        assert isinstance(m["fn"], int)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def _make_labels(self) -> list[int]:
        return [0] * 600 + [1] * 400

    def test_random_baseline_returns_all_keys(self):
        y = self._make_labels()
        m = baseline_random(y)
        assert "accuracy" in m and "f1" in m

    def test_random_baseline_reproducible(self):
        y = self._make_labels()
        m1 = baseline_random(y, seed=99)
        m2 = baseline_random(y, seed=99)
        assert m1 == m2

    def test_majority_baseline_always_predicts_majority(self):
        y = self._make_labels()  # majority = 0
        m = baseline_majority(y)
        # predicts all 0 → TP=0, FP=0, FN=400, TN=600
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 400
        assert m["tn"] == 600

    def test_majority_baseline_accuracy(self):
        y = self._make_labels()
        m = baseline_majority(y)
        # 600 correct out of 1000
        assert m["accuracy"] == pytest.approx(0.6)

    def test_majority_baseline_f1_zero_for_minority(self):
        # When majority is always predicted, no true positives for minority
        y = self._make_labels()
        m = baseline_majority(y)
        assert m["recall"] == pytest.approx(0.0)

    def test_random_baseline_counts_sum_to_total(self):
        y = self._make_labels()
        m = baseline_random(y)
        total = m["tp"] + m["tn"] + m["fp"] + m["fn"]
        assert total == len(y)
