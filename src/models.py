"""CV trên development, threshold trên validation, refit trên toàn outer Train."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import joblib
from joblib import Memory, parallel_config
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import MODEL_LABELS, classification_metrics, tune_decision_threshold
from src.feature_selection import HybridFeatureSelector
from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    n_iter: int = 6
    cv_folds: int = 3
    search_jobs: int = 2
    model_jobs: int = 2
    validation_size: float = 0.2
    min_precision: float = 0.2
    shap_samples: int = 300

    def validate(self):
        if self.n_iter < 1 or self.cv_folds < 2 or self.search_jobs < 1 or self.model_jobs < 1:
            raise ValueError("Iterations/jobs phải dương; CV phải ít nhất 2 fold.")
        if not 0 < self.validation_size < 0.5 or not 0 < self.min_precision <= 1 or self.shap_samples < 1:
            raise ValueError("validation_size/min_precision/shap_samples không hợp lệ.")


def build_model_pipeline(model_name, mappings, config: ExperimentConfig, memory=None):
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000, random_state=config.random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=160, min_samples_leaf=8,
            n_jobs=config.model_jobs, random_state=config.random_state,
        ),
        "lightgbm": LGBMClassifier(
            is_unbalance=True, objective="binary", n_jobs=config.model_jobs,
            random_state=config.random_state, verbosity=-1, importance_type="gain",
            deterministic=True, force_col_wise=True, subsample_freq=1,
        ),
    }
    if model_name not in estimators:
        raise ValueError(f"Thuật toán không được hỗ trợ: {model_name}")
    return Pipeline([
        ("preprocessing", build_preprocessor(id_mappings=mappings)),
        ("feature_selection", HybridFeatureSelector(random_state=config.random_state, n_jobs=config.model_jobs)),
        ("classifier", estimators[model_name]),
    ], memory=memory)


def parameter_space(model_name):
    common = {"feature_selection__k": [40, 80, 120, "all"]}
    spaces = {
        "logistic_regression": {"classifier__C": [0.01, 0.03, 0.1, 0.3, 1., 3., 10.]},
        "random_forest": {
            "classifier__n_estimators": [160, 240], "classifier__max_depth": [10, 16, None],
            "classifier__min_samples_leaf": [4, 12, 25], "classifier__max_features": ["sqrt", 0.5],
            "classifier__max_samples": [0.8, 1.0],
        },
        "lightgbm": {
            "classifier__n_estimators": [200, 400], "classifier__learning_rate": [0.03, 0.07],
            "classifier__num_leaves": [15, 31], "classifier__max_depth": [-1, 8],
            "classifier__min_child_samples": [30, 80], "classifier__reg_lambda": [1., 5.],
            "classifier__colsample_bytree": [0.8, 1.0], "classifier__subsample": [0.8, 1.0],
        },
    }
    return {**common, **spaces[model_name]}


def choose_model_on_validation(records: list[dict]) -> str:
    """Không nhận Test; ưu tiên đạt sàn precision, rồi F2/Recall/AP validation."""
    return max(records, key=lambda record: (
        record["validation_tuned"]["precision_constraint_met"],
        record["validation_tuned"]["F2"], record["validation_tuned"]["Recall"],
        record["validation_tuned"]["Average_Precision"],
    ))["model"]


def train_models(X_train, y_train, mappings, output_dir, config=None, train_ids=None):
    """Không có tham số X_test/y_test: test không thể tham gia việc lựa chọn."""
    config = config or ExperimentConfig()
    config.validate()
    output_dir = Path(output_dir)
    tables, artifacts = output_dir / "tables", output_dir / "artifacts"
    tables.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    development_positions, validation_positions = train_test_split(
        np.arange(len(X_train)), test_size=config.validation_size,
        stratify=np.asarray(y_train), random_state=config.random_state,
    )
    X_development, X_validation = X_train.iloc[development_positions], X_train.iloc[validation_positions]
    y_development, y_validation = y_train.iloc[development_positions], y_train.iloc[validation_positions]
    if int(y_development.value_counts().min()) < config.cv_folds:
        raise ValueError("Không đủ nhãn mỗi lớp để phân tầng CV.")
    membership = pd.DataFrame({
        "outer_train_position": np.concatenate([development_positions, validation_positions]),
        "partition": ["development"] * len(development_positions) + ["threshold_validation"] * len(validation_positions),
    })
    if train_ids is not None:
        membership["patient_nbr"] = train_ids.iloc[membership.outer_train_position].patient_nbr.to_numpy()
        if membership.patient_nbr.duplicated().any():
            raise ValueError("Trùng bệnh nhân trong split nội bộ.")
    membership.to_csv(tables / "internal_split_membership.csv", index=False)
    print(f"Development: {len(X_development):,}; threshold validation: {len(X_validation):,}.", flush=True)
    validation_predictions = pd.DataFrame({"outer_train_position": validation_positions, "y_true": np.asarray(y_validation)})
    memory = Memory(output_dir / ".cache" / "model_pipeline", verbose=0)
    records, fitted_models = [], {}
    scoring = {"average_precision": "average_precision", "roc_auc": "roc_auc", "recall": "recall",
               "f2": make_scorer(fbeta_score, beta=2, zero_division=0)}
    for model_name in MODEL_LABELS:
        started = time.perf_counter()
        print(f"[{MODEL_LABELS[model_name]}] RandomizedSearchCV: {config.n_iter} candidates × {config.cv_folds} folds...", flush=True)
        pipeline = build_model_pipeline(model_name, mappings, config, memory)
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        search = RandomizedSearchCV(
            pipeline, parameter_space(model_name), n_iter=config.n_iter, cv=cv,
            scoring=scoring, refit="average_precision", random_state=config.random_state,
            n_jobs=config.search_jobs, pre_dispatch=config.search_jobs, error_score="raise", return_train_score=False,
        )
        # Hạn chế thread BLAS trong worker, tránh nhân số luồng ngoài ý muốn.
        with parallel_config(backend="loky", inner_max_num_threads=config.model_jobs):
            search.fit(X_development, y_development)
        cv_results = pd.DataFrame(search.cv_results_)
        for column in cv_results.filter(regex=r"^param_").columns:
            cv_results[column] = cv_results[column].astype(str)
        cv_results.to_csv(tables / f"cv_results_{model_name}.csv", index=False)
        scores = search.best_estimator_.predict_proba(X_validation)[:, 1]
        validation_predictions[model_name] = scores
        threshold, curve, validation_tuned = tune_decision_threshold(y_validation, scores, config.min_precision)
        curve.to_csv(tables / f"threshold_curve_{model_name}.csv", index=False)
        # Khóa tham số/ngưỡng trước khi refit trên full outer Train, không có Test.
        final_model = clone(search.best_estimator_).set_params(memory=None)
        final_model.fit(X_train, y_train)
        preprocessor = final_model.named_steps["preprocessing"]
        selector = final_model.named_steps["feature_selection"]
        ranking = selector.ranking_table(preprocessor.get_feature_names_out())
        ranking.to_csv(tables / f"feature_selection_{model_name}.csv", index=False)
        ranking.loc[ranking.selected].to_csv(tables / f"selected_features_{model_name}.csv", index=False)
        artifact_path = artifacts / f"{model_name}.joblib"
        joblib.dump(final_model, artifact_path, compress=3)
        record = {
            "model": model_name, "label": MODEL_LABELS[model_name], "threshold": threshold,
            "cv_best_average_precision": float(search.best_score_),
            "cv_best_roc_auc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
            "best_params": search.best_params_, "selected_features": int(selector.get_support().sum()),
            "preprocessed_features": int(selector.n_features_in_),
            "validation_default": classification_metrics(y_validation, scores, 0.5),
            "validation_tuned": validation_tuned, "artifact_path": str(artifact_path.resolve()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        records.append(record)
        fitted_models[model_name] = final_model
        print(f"[{record['label']}] k={record['selected_features']}; validation threshold={threshold:.2f}; "
              f"Recall={validation_tuned['Recall']:.3f}; Precision={validation_tuned['Precision']:.3f}; "
              f"F2={validation_tuned['F2']:.3f}; {record['elapsed_seconds']:.1f}s.", flush=True)
    validation_predictions.to_csv(tables / "validation_predictions.csv", index=False)
    chosen = choose_model_on_validation(records)
    validation_rows = [{"model": record["model"], "operating_point": point,
                        **record[f"validation_{point}"]} for record in records for point in ("default", "tuned")]
    pd.DataFrame(validation_rows).to_csv(tables / "validation_metrics.csv", index=False)
    summary = {
        "config": asdict(config), "selection_partition": "internal_validation_within_outer_train",
        "development_rows": len(X_development), "threshold_validation_rows": len(X_validation),
        "outer_train_rows": len(X_train), "outer_train_positive_rate": float(np.mean(y_train)),
        "cv_refit_metric": "average_precision", "chosen_model": chosen, "models": records,
        "class_imbalance": "LR/RF class_weight=balanced; LightGBM is_unbalance=True; no SMOTE",
        "probability_note": "Class-weighted predict_proba scores are not calibrated clinical risk probabilities.",
        "selection_rule": "Precision floor on validation, then highest validation F2, Recall, Average Precision; never Test.",
    }
    (tables / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_record = next(record for record in records if record["model"] == chosen)
    (artifacts / "best_model_metadata.json").write_text(json.dumps({
        **best_record, "selection_partition": summary["selection_partition"],
        "probability_note": summary["probability_note"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return fitted_models, summary
