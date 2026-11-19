"""Dự đoán tái nhập viện: EDA, tiền xử lý, chọn biến, ba mô hình và đánh giá."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import inspect
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.preprocessing import (
    FIXED_DROP_COLUMNS, RANDOM_STATE, build_preprocessor, get_numeric_transform_report,
    load_diabetes_data, load_id_mappings, prepare_cohort, split_cohort,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _configure_console():
    """Giữ log tiếng Việt hoạt động cả khi gọi API từ Python -c trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and stream.encoding.lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preprocessing(data_dir=PROJECT_ROOT, output_dir=None, make_plots=True) -> dict:
    """Hàm dùng chung cho CLI và notebook; giữ nguyên hai CSV nguồn."""
    _configure_console()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir or PROJECT_ROOT / "outputs").resolve()
    for directory in ("processed", "artifacts", "tables", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    data_path, mapping_path = data_dir / "diabetic_data.csv", data_dir / "IDS_mapping.csv"
    raw = load_diabetes_data(data_path)
    mappings = load_id_mappings(mapping_path)
    cohort, audit = prepare_cohort(raw)
    splits = split_cohort(cohort)
    preprocessor = build_preprocessor(id_mappings=mappings)
    train_matrix = sparse.csr_matrix(preprocessor.fit_transform(splits.X_train, splits.y_train), dtype=np.float32)
    test_matrix = sparse.csr_matrix(preprocessor.transform(splits.X_test), dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out()
    if not np.isfinite(train_matrix.data).all() or not np.isfinite(test_matrix.data).all():
        raise RuntimeError("Ma trận sau tiền xử lý còn NaN/vô hạn.")
    if train_matrix.shape[1] != test_matrix.shape[1] or train_matrix.shape[1] != len(feature_names):
        raise RuntimeError("Schema Train/Test không khớp.")
    for partition in ("train", "test"):
        X, y = getattr(splits, f"X_{partition}"), getattr(splits, f"y_{partition}")
        X.to_csv(output_dir / "processed" / f"X_{partition}_raw.csv", index=False, encoding="utf-8-sig")
        y.to_csv(output_dir / "processed" / f"y_{partition}.csv", index=False)
    sparse.save_npz(output_dir / "processed" / "X_train_preprocessed.npz", train_matrix)
    sparse.save_npz(output_dir / "processed" / "X_test_preprocessed.npz", test_matrix)
    membership = pd.concat([
        splits.train_ids.assign(partition="train"), splits.test_ids.assign(partition="test"),
    ])
    membership.to_csv(output_dir / "tables" / "split_membership.csv", index=False)
    joblib.dump(preprocessor, output_dir / "artifacts" / "preprocessor.joblib", compress=3)
    pd.DataFrame({"feature": feature_names}).to_csv(output_dir / "tables" / "feature_names.csv", index=False)
    transforms = get_numeric_transform_report(preprocessor, splits.X_train)
    transforms.to_csv(output_dir / "tables" / "training_numeric_transforms.csv", index=False)
    distribution = pd.DataFrame([
        {"partition": partition, "rows": len(getattr(splits, f"y_{partition}")),
         "positive_count": int(getattr(splits, f"y_{partition}").sum()),
         "positive_rate": float(getattr(splits, f"y_{partition}").mean())}
        for partition in ("train", "test")
    ])
    distribution.to_csv(output_dir / "tables" / "split_distribution.csv", index=False)
    figures = {}
    if make_plots:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".cache" / "matplotlib"))
        from src.eda import create_eda_artifacts
        figures = create_eda_artifacts(raw, cohort, splits.X_train, preprocessor, output_dir)
    summary = {
        **audit, "random_state": RANDOM_STATE, "test_size": 0.2,
        "split_distribution": distribution.to_dict(orient="records"),
        "patient_overlap": len(set(splits.train_ids.patient_nbr) & set(splits.test_ids.patient_nbr)),
        "train_matrix_shape": list(train_matrix.shape), "test_matrix_shape": list(test_matrix.shape),
        "all_transformed_values_finite": True, "fixed_dropped_columns": list(FIXED_DROP_COLUMNS),
        "additional_columns_dropped_on_train": preprocessor.named_steps["column_filter"].dropped_reasons_,
        "log_transformed_columns": transforms.loc[transforms.log1p, "feature"].tolist(),
        "figures": figures,
        "source_sha256": {data_path.name: _sha256(data_path), mapping_path.name: _sha256(mapping_path)},
        "python": platform.python_version(),
        "package_versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "joblib"]},
        "cv_note": "For CV, use X_train_raw.csv and a fresh preprocessor within each training fold; NPZ matrices are preprocessing checks, not CV inputs.",
        "completed_steps": [1, 2, 3],
    }
    (output_dir / "tables" / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Hoàn tất Bước 1–3. Cohort: {len(cohort):,} bệnh nhân.")
    print(distribution.to_string(index=False))
    print(f"Ma trận Train: {train_matrix.shape}; Test: {test_matrix.shape}; giao bệnh nhân: 0.")
    print(f"Kết quả: {output_dir}")
    return summary


def experiment_provenance(data_dir, config):
    """Chỉ tái sử dụng mô hình khi dữ liệu, mã huấn luyện và cấu hình khớp."""
    from src.evaluate import classification_metrics, tune_decision_threshold
    source_hashes = {name: _sha256(Path(data_dir) / name) for name in ["diabetic_data.csv", "IDS_mapping.csv"]}
    payload = {
        "source_sha256": source_hashes, "config": asdict(config),
        "training_code_sha256": {name: _sha256(PROJECT_ROOT / "src" / name)
                                 for name in ["preprocessing.py", "feature_selection.py", "models.py"]},
        "threshold_and_metric_code": inspect.getsource(tune_decision_threshold) + inspect.getsource(classification_metrics),
        "package_versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "shap"]},
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return signature, payload


def run_modeling(data_dir=PROJECT_ROOT, output_dir=None, config=None, retrain=False, report_only=False):
    _configure_console()
    from src.evaluate import create_evaluation_artifacts, write_experiment_report
    from src.models import ExperimentConfig, train_models

    config = config or ExperimentConfig()
    config.validate()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir or PROJECT_ROOT / "outputs").resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".cache" / "matplotlib"))
    for directory in ("tables", "artifacts", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    signature, provenance = experiment_provenance(data_dir, config)
    training_path = output_dir / "tables" / "training_summary.json"
    result_path = output_dir / "tables" / "experiment_summary.json"
    cached = json.loads(training_path.read_text(encoding="utf-8")) if training_path.is_file() else {}
    reusable = not retrain and cached.get("signature") == signature and all(
        Path(record["artifact_path"]).is_file() for record in cached.get("models", [])
    ) and len(cached.get("models", [])) == 3
    if report_only and not reusable:
        raise ValueError("Chưa có đủ mô hình phù hợp dữ liệu/cấu hình/mã hiện tại. Hãy chạy --stage all trước.")
    if reusable and not report_only and result_path.is_file():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        required_files = [output_dir / "tables" / "model_comparison.csv", output_dir / "report.md",
                          output_dir / "artifacts" / "shap_lightgbm.npz", *map(Path, completed.get("figures", {}).values())]
        if completed.get("signature") == signature and all(path.is_file() for path in required_files):
            print("Đã có kết quả đủ ba mô hình khớp dữ liệu/mã/cấu hình; tái sử dụng. Dùng --retrain để huấn luyện lại.", flush=True)
            return completed
    raw = load_diabetes_data(data_dir / "diabetic_data.csv")
    mappings = load_id_mappings(data_dir / "IDS_mapping.csv")
    cohort, audit = prepare_cohort(raw)
    splits = split_cohort(cohort, random_state=config.random_state)
    if reusable:
        training = cached
        models = {record["model"]: joblib.load(record["artifact_path"]) for record in cached["models"]}
        print("Dùng các mô hình đã khóa; tạo lại bảng/biểu đồ mà không tuning từ Test.", flush=True)
    else:
        models, training = train_models(splits.X_train, splits.y_train, mappings, output_dir, config, splits.train_ids)
        training.update({"signature": signature, "provenance": provenance, "cohort_audit": audit})
        training_path.write_text(json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã khóa mô hình/ngưỡng theo validation. Đang đánh giá Test và tạo biểu đồ/SHAP...", flush=True)
    comparison, figures = create_evaluation_artifacts(models, training, splits.X_test, splits.y_test, splits.X_train, output_dir)
    write_experiment_report(training, comparison, figures, audit, output_dir)
    chosen = training["chosen_model"]
    result = {
        "signature": signature, "completed_steps": [1, 2, 3, 4, 5, 6],
        "cohort_audit": audit, "config": asdict(config), "provenance": provenance,
        "chosen_model": chosen, "selection_partition": training["selection_partition"],
        "models": training["models"], "figures": figures,
        "test_metrics": comparison.to_dict(orient="records"),
        "report_path": str((output_dir / "report.md").resolve()),
        "comparison_path": str((output_dir / "tables" / "model_comparison.csv").resolve()),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison[["label", "operating_point", "threshold", "Recall", "Precision", "F1", "F2", "ROC_AUC", "PR_AUC"]].to_string(index=False), flush=True)
    print(f"Mô hình chọn trên validation: {chosen}. Báo cáo: {output_dir / 'report.md'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT, help="Thư mục chứa hai CSV nguồn.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--no-plots", action="store_true", help="Bỏ tạo ảnh khi chỉ muốn kiểm tra ma trận.")
    parser.add_argument("--stage", choices=["all", "preprocessing", "report"], default="all")
    parser.add_argument("--iterations", type=int, default=6, help="Số cấu hình RandomizedSearchCV mỗi thuật toán.")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2, help="Số CV worker; mỗi mô hình dùng tối đa 2 thread.")
    parser.add_argument("--min-precision", type=float, default=.20, help="Sàn Precision validation để chọn threshold.")
    parser.add_argument("--shap-samples", type=int, default=300)
    parser.add_argument("--retrain", action="store_true", help="Bỏ cache mô hình và thực hiện lại CV.")
    args = parser.parse_args()
    if args.stage != "report":
        run_preprocessing(args.data_dir, args.output_dir, make_plots=not args.no_plots)
    if args.stage != "preprocessing":
        from src.models import ExperimentConfig
        config = ExperimentConfig(n_iter=args.iterations, cv_folds=args.cv_folds, search_jobs=args.jobs,
                                  min_precision=args.min_precision, shap_samples=args.shap_samples)
        run_modeling(args.data_dir, args.output_dir, config, retrain=args.retrain, report_only=args.stage == "report")


if __name__ == "__main__":
    _configure_console()
    main()
