"""Kiểm tra các quy tắc có thể gây sai kết quả hoặc leakage."""

import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from src.preprocessing import (
    ACTIVE_DRUG_COLUMNS, ADMIN_COLUMNS, DIAG_COLUMNS, DRUG_COLUMNS, ID_COLUMNS,
    NUMERIC_COLUMNS, REQUIRED_COLUMNS, TARGET, ClinicalFeatureEngineer,
    SkewAwareLogTransformer, TrainOnlyColumnFilter, build_preprocessor, group_icd9,
    load_diabetes_data, load_id_mappings, prepare_cohort, split_cohort,
)


def synthetic_data(n=40):
    records = []
    for index in range(n):
        row = {column: "No" for column in REQUIRED_COLUMNS}
        row.update({column: "No" for column in DRUG_COLUMNS})
        row.update({column: 0 for column in NUMERIC_COLUMNS})
        row.update({column: "250.83" for column in DIAG_COLUMNS})
        row.update({column: "1" for column in ADMIN_COLUMNS})
        row.update({
            "encounter_id": str(1000 + index), "patient_nbr": str(100 + index),
            "race": "Caucasian" if index % 2 else np.nan, "gender": "Female" if index % 2 else "Male",
            "age": f"[{(index % 9 + 1) * 10}-{(index % 9 + 2) * 10})", "weight": "?",
            "payer_code": np.nan, "medical_specialty": "InternalMedicine" if index % 3 else np.nan,
            "time_in_hospital": index % 14 + 1, "num_medications": index % 10 + 1,
            "num_lab_procedures": 20 + index, "num_procedures": index % 4,
            "number_inpatient": index % 3, "number_diagnoses": index % 5 + 1,
            "max_glu_serum": "None" if index % 3 else ">200",
            "A1Cresult": "Norm" if index % 2 else "None",
            "insulin": ["No", "Steady", "Down", "Up"][index % 4],
            "change": "Ch" if index % 4 in (2, 3) else "No", "diabetesMed": "Yes",
            "readmitted": "<30" if index % 4 == 0 else "NO",
        })
        records.append(row)
    return pd.DataFrame(records)


class CohortTests(unittest.TestCase):
    def test_reading_preserves_not_measured_but_normalizes_missing(self):
        raw = synthetic_data()
        raw.loc[0, "gender"] = "Unknown/Invalid"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            raw.to_csv(path, index=False)
            loaded = load_diabetes_data(path)
        self.assertEqual(loaded.loc[0, "max_glu_serum"], ">200")
        self.assertEqual(loaded.loc[1, "max_glu_serum"], "None")
        self.assertTrue(pd.isna(loaded.loc[0, "weight"]))
        self.assertTrue(pd.isna(loaded.loc[0, "gender"]))

    def test_earliest_proxy_selected_before_exclusion(self):
        raw = synthetic_data()
        raw.loc[0, "discharge_disposition_id"] = "11"
        later = raw.iloc[[0]].copy()
        later["encounter_id"] = "9999"
        later["discharge_disposition_id"] = "1"
        cohort, audit = prepare_cohort(pd.concat([later, raw], ignore_index=True))
        self.assertNotIn("100", cohort.patient_nbr.to_list())
        self.assertEqual(audit["repeated_encounters_removed"], 1)
        self.assertEqual(audit["death_hospice_removed_after_deduplication"], 1)

    def test_all_six_excluded_dispositions(self):
        raw = synthetic_data()
        raw.loc[:5, "discharge_disposition_id"] = ["11", "13", "14", "19", "20", "21"]
        cohort, audit = prepare_cohort(raw)
        self.assertEqual(audit["death_hospice_removed_after_deduplication"], 6)
        self.assertEqual(len(cohort), len(raw) - 6)

    def test_target_stratification_and_patient_isolation(self):
        raw = synthetic_data(100)
        raw.loc[1, "readmitted"] = ">30"
        cohort, _ = prepare_cohort(raw)
        splits = split_cohort(cohort)
        self.assertEqual((len(splits.X_train), len(splits.X_test)), (80, 20))
        self.assertFalse(set(splits.train_ids.patient_nbr) & set(splits.test_ids.patient_nbr))
        self.assertFalse(set((*ID_COLUMNS, "readmitted", TARGET)) & set(splits.X_train.columns))
        self.assertEqual(splits.y_train.mean(), splits.y_test.mean())
        again = split_cohort(cohort)
        pd.testing.assert_frame_equal(splits.X_train, again.X_train)
        self.assertEqual(cohort.loc[cohort.patient_nbr.eq("101"), TARGET].iloc[0], 0)

    def test_invalid_target_and_duplicate_encounter_rejected(self):
        raw = synthetic_data()
        raw.loc[0, "readmitted"] = np.nan
        with self.assertRaises(ValueError):
            prepare_cohort(raw)
        raw = synthetic_data()
        raw.loc[0, "encounter_id"] = raw.loc[1, "encounter_id"]
        with self.assertRaises(ValueError):
            prepare_cohort(raw)

    def test_mapping_file_has_three_sections(self):
        mapping = load_id_mappings(Path(__file__).resolve().parents[1] / "IDS_mapping.csv")
        self.assertEqual(set(mapping), set(ADMIN_COLUMNS))
        self.assertEqual(mapping["discharge_disposition_id"]["11"], "Expired")
        self.assertEqual(mapping["admission_source_id"]["7"], "Emergency Room")


class ClinicalFeatureTests(unittest.TestCase):
    def test_icd9_boundaries_decimals_and_missing(self):
        cases = {
            "250.83": "Diabetes", "459.9": "Circulatory", "785.2": "Circulatory",
            "460": "Respiratory", "786": "Respiratory", "579.9": "Digestive",
            "787": "Digestive", "580": "Genitourinary", "788": "Genitourinary",
            "710": "Musculoskeletal", "739.9": "Musculoskeletal", "140": "Neoplasms",
            "239.9": "Neoplasms", "800": "Injury", "999": "Injury",
            "V45.81": "Other", "E849.0": "Other", "780": "Other",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(group_icd9(value), expected)
        self.assertTrue(pd.isna(group_icd9(np.nan)))
        self.assertTrue(pd.isna(group_icd9("?")))

    def test_drug_direction_and_not_prescribed_are_distinct(self):
        X = synthetic_data().drop(columns=[*ID_COLUMNS, "readmitted"])
        engineered = ClinicalFeatureEngineer().fit_transform(X)
        self.assertEqual(engineered.loc[0, "insulin"], "Steady")
        self.assertEqual(engineered.loc[1, "insulin"], "Steady")
        self.assertEqual(engineered.loc[0, "insulin_prescribed"], 0)
        self.assertEqual(engineered.loc[1, "insulin_prescribed"], 1)
        self.assertEqual(engineered.loc[2, "n_medications_changed"], 1)
        self.assertEqual(engineered.loc[2, "n_medications_decreased"], 1)
        self.assertEqual(engineered.loc[3, "n_medications_increased"], 1)
        self.assertEqual(engineered.loc[2, "total_prior_visits"], 2)
        self.assertNotIn("weight", engineered)
        self.assertNotIn("diag_1", engineered)

    def test_not_measured_distinct_from_normal(self):
        X = synthetic_data().drop(columns=[*ID_COLUMNS, "readmitted"])
        engineered = ClinicalFeatureEngineer().fit_transform(X)
        self.assertTrue(pd.isna(engineered.loc[0, "A1Cresult"]))
        self.assertEqual(engineered.loc[0, "A1Cresult_measured"], 0)
        self.assertEqual(engineered.loc[1, "A1Cresult"], "Norm")
        self.assertEqual(engineered.loc[1, "A1Cresult_measured"], 1)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        cohort, _ = prepare_cohort(synthetic_data(80))
        self.splits = split_cohort(cohort)

    def test_transform_does_not_learn_holdout_statistics_or_categories(self):
        pipeline = build_preprocessor()
        pipeline.fit(self.splits.X_train, self.splits.y_train)
        numeric = pipeline.named_steps["columns"].named_transformers_["numeric"]
        median_before = numeric.named_steps["imputer"].statistics_.copy()
        scale_before = numeric.named_steps["scaler"].scale_.copy()
        feature_names = pipeline.get_feature_names_out().copy()
        holdout = self.splits.X_test.copy()
        holdout["num_lab_procedures"] = 999999
        holdout["race"] = "UnseenRaceOnlyOnTest"
        transformed = sparse.csr_matrix(pipeline.transform(holdout))
        np.testing.assert_array_equal(median_before, numeric.named_steps["imputer"].statistics_)
        np.testing.assert_array_equal(scale_before, numeric.named_steps["scaler"].scale_)
        np.testing.assert_array_equal(feature_names, pipeline.get_feature_names_out())
        self.assertEqual(transformed.shape[1], len(feature_names))
        self.assertTrue(np.isfinite(transformed.data).all())
        self.assertFalse(any("UnseenRaceOnlyOnTest" in name for name in feature_names))

    def test_filter_and_log_decisions_come_from_train(self):
        train = pd.DataFrame({"constant": [1, 1, 1], "varying": [1, 2, 3], "missing": [np.nan] * 3})
        column_filter = TrainOnlyColumnFilter().fit(train)
        holdout = pd.DataFrame({"constant": [2], "varying": [100], "missing": [4]})
        self.assertEqual(column_filter.transform(holdout).columns.to_list(), ["varying"])
        log = SkewAwareLogTransformer().fit([[0], [1], [2], [3]])
        np.testing.assert_array_equal(log.transform([[999]]), [[999]])

    def test_pipeline_is_cloneable_serializable_and_produces_finite_matrices(self):
        pipeline = clone(build_preprocessor())
        train = sparse.csr_matrix(pipeline.fit_transform(self.splits.X_train, self.splits.y_train))
        test = sparse.csr_matrix(pipeline.transform(self.splits.X_test))
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[1], len(pipeline.get_feature_names_out()))
        self.assertTrue(np.isfinite(train.data).all())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pipeline.joblib"
            joblib.dump(pipeline, path)
            restored = sparse.csr_matrix(joblib.load(path).transform(self.splits.X_test))
        self.assertEqual((restored != test).nnz, 0)

    def test_target_cannot_be_passed_to_preprocessor(self):
        X = self.splits.X_train.assign(readmitted="<30")
        with self.assertRaises(ValueError):
            build_preprocessor().fit(X)

    def test_each_engineered_column_assigned_to_exactly_one_branch(self):
        pipeline = build_preprocessor().fit(self.splits.X_train)
        assignments = [column
            for name, transformer, columns in pipeline.named_steps["columns"].transformers_
            if name != "remainder" for column in columns]
        self.assertEqual(len(assignments), len(set(assignments)))
        self.assertEqual(set(assignments), set(pipeline.named_steps["column_filter"].kept_columns_))


if __name__ == "__main__":
    unittest.main()
