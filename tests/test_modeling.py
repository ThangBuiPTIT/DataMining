"""Kiểm tra lựa chọn trên Train/validation, metric, threshold và model pipeline."""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import clone

from src.evaluate import classification_metrics, short_feature, tune_decision_threshold
from src.feature_selection import HybridFeatureSelector
from src.models import ExperimentConfig, choose_model_on_validation, train_models
from src.preprocessing import prepare_cohort, split_cohort
from test_preprocessing import synthetic_data


class MetricAndThresholdTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 1, 0, 0, 0, 0])
        self.scores = np.array([.55, .2, .5, .3, .1, .05])

    def test_metrics_match_hand_calculated_counts_and_f2(self):
        result = classification_metrics(self.y, self.scores, .2)
        self.assertEqual((result["TN"], result["FP"], result["FN"], result["TP"]), (2, 2, 0, 2))
        self.assertEqual(result["Recall"], 1)
        self.assertEqual(result["Precision"], .5)
        self.assertAlmostEqual(result["F1"], 2 / 3)
        self.assertAlmostEqual(result["F2"], 5 / 6)
        self.assertAlmostEqual(result["ROC_AUC"], .75)
        self.assertAlmostEqual(result["Average_Precision"], .75)

    def test_threshold_maximizes_f2_with_precision_constraint(self):
        threshold, curve, result = tune_decision_threshold(
            self.y, self.scores, min_precision=.5, thresholds=[.1, .2, .3, .5, .6])
        self.assertEqual(threshold, .2)
        self.assertTrue(result["precision_constraint_met"])
        self.assertGreaterEqual(result["Precision"], .5)
        self.assertEqual(curve.loc[curve.precision_constraint_met, "F2"].max(), result["F2"])

    def test_infeasible_precision_is_reported_instead_of_hidden(self):
        threshold, _, result = tune_decision_threshold(
            self.y, self.scores, min_precision=.9, thresholds=[.1, .2, .3, .5, .6])
        self.assertEqual(threshold, .2)
        self.assertFalse(result["precision_constraint_met"])

    def test_auc_is_independent_of_operating_threshold(self):
        default = classification_metrics(self.y, self.scores, .5)
        tuned = classification_metrics(self.y, self.scores, .2)
        for metric in ("ROC_AUC", "PR_AUC", "Average_Precision"):
            self.assertEqual(default[metric], tuned[metric])
        self.assertGreater(tuned["Recall"], default["Recall"])

    def test_invalid_scores_labels_or_grid_are_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics(self.y, [np.nan] * 6)
        with self.assertRaises(ValueError):
            classification_metrics([0, 1.5], [.1, .2])
        with self.assertRaises(ValueError):
            tune_decision_threshold(self.y, self.scores, thresholds=[.09, .5])

    def test_model_choice_uses_validation_and_prioritizes_precision_floor(self):
        records = [
            {"model": "logistic_regression", "validation_tuned": {
                "precision_constraint_met": True, "F2": .4, "Recall": .5, "Average_Precision": .2}, "test_f2": .1},
            {"model": "random_forest", "validation_tuned": {
                "precision_constraint_met": False, "F2": .8, "Recall": 1., "Average_Precision": .3}, "test_f2": 1.},
        ]
        self.assertEqual(choose_model_on_validation(records), "logistic_regression")

    def test_long_administrative_labels_keep_distinct_codes(self):
        descriptions = {"discharge_disposition_id": {
            "2": "Discharged/transferred to another short term hospital",
            "22": "Discharged/transferred to another rehab facility"}}
        labels = [short_feature(f"nominal__discharge_disposition_id_{value}", 43, descriptions)
                  for value in descriptions["discharge_disposition_id"].values()]
        self.assertNotEqual(*labels)
        self.assertIn("discharge=2:", labels[0])
        self.assertIn("discharge=22:", labels[1])


class FeatureSelectionTests(unittest.TestCase):
    def test_both_rankings_find_strong_signal_without_refitting_on_holdout(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(500, 8))
        y = (matrix[:, 0] > 0).astype(int)
        selector = clone(HybridFeatureSelector(k=3, n_estimators=30, n_jobs=1)).fit(sparse.csr_matrix(matrix[:400]), y[:400])
        self.assertEqual(np.argmax(selector.anova_scores_), 0)
        self.assertEqual(np.argmax(selector.model_importances_), 0)
        self.assertIn(0, selector.get_support(indices=True))
        support_before = selector.get_support().copy()
        transformed = selector.transform(sparse.csr_matrix(matrix[400:] * 100))
        np.testing.assert_array_equal(selector.get_support(), support_before)
        self.assertEqual(transformed.shape, (100, 3))
        self.assertEqual(len(selector.ranking_table([f"f{i}" for i in range(8)])), 8)

    def test_k_larger_than_available_columns_and_all_are_supported(self):
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(160, 4))
        y = np.tile([0, 1], 80)
        for k in (100, "all"):
            selector = HybridFeatureSelector(k=k, n_jobs=1, n_estimators=3).fit(matrix, y)
            self.assertEqual(selector.transform(matrix).shape[1], 4)


class TrainingIntegrationTests(unittest.TestCase):
    def test_three_models_tune_refit_serialize_and_keep_internal_groups_separate(self):
        cohort, _ = prepare_cohort(synthetic_data(240))
        splits = split_cohort(cohort)
        config = ExperimentConfig(n_iter=1, cv_folds=2, search_jobs=1, model_jobs=1, shap_samples=10)
        with tempfile.TemporaryDirectory() as temp_dir:
            models, summary = train_models(splits.X_train, splits.y_train, None, temp_dir, config, splits.train_ids)
            self.assertEqual(set(models), {"logistic_regression", "random_forest", "lightgbm"})
            self.assertEqual(summary["development_rows"] + summary["threshold_validation_rows"], len(splits.X_train))
            import pandas as pd
            membership = pd.read_csv(Path(temp_dir) / "tables" / "internal_split_membership.csv")
            self.assertTrue(membership.patient_nbr.is_unique)
            self.assertEqual(set(membership.partition), {"development", "threshold_validation"})
            for record in summary["models"]:
                self.assertTrue(.1 <= record["threshold"] <= .6)
                restored = joblib.load(record["artifact_path"])
                scores = restored.predict_proba(splits.X_test)[:, 1]
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(len(scores), len(splits.X_test))
                np.testing.assert_allclose(scores, models[record["model"]].predict_proba(splits.X_test)[:, 1])


if __name__ == "__main__":
    unittest.main()
