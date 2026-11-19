"""EDA mô tả và biểu đồ cho báo cáo Bước 1–3; không chọn biến từ tập Test."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import DIAG_COLUMNS, NUMERIC_COLUMNS, TARGET, get_numeric_transform_report


def create_eda_artifacts(raw, cohort, X_train, preprocessor, output_dir) -> dict[str, str]:
    """Raw chỉ dùng thống kê mô tả; mọi EDA đặc trưng mô hình dùng Train."""
    output_dir = Path(output_dir)
    figures, tables = output_dir / "figures", output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans", palette="colorblind")
    paths = {}

    def save_figure(name, fig):
        fig.tight_layout()
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path.resolve())

    schema = pd.DataFrame({
        "feature": raw.columns,
        "dtype": raw.dtypes.astype(str).to_numpy(),
        "missing_count": raw.isna().sum().to_numpy(),
        "missing_percent": raw.isna().mean().mul(100).to_numpy(),
        "unique_non_missing": raw.nunique(dropna=True).to_numpy(),
    }).sort_values("missing_percent", ascending=False)
    schema.to_csv(tables / "raw_schema_and_missing.csv", index=False, encoding="utf-8-sig")
    raw_counts = raw["readmitted"].value_counts().reindex(["NO", ">30", "<30"], fill_value=0)
    raw_counts.rename_axis("readmitted").rename("count").to_csv(tables / "raw_class_counts.csv")
    cohort[TARGET].value_counts().sort_index().rename_axis(TARGET).rename("count").to_csv(
        tables / "cohort_class_counts.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, labels, counts, title in (
        (axes[0], raw_counts.index, raw_counts.values, "Nhãn ban đầu: toàn bộ dữ liệu"),
        (axes[1], ["0: NO / >30", "1: <30"], cohort[TARGET].value_counts().reindex([0, 1], fill_value=0).values,
         "Mục tiêu nhị phân: sau lọc cohort"),
    ):
        bars = ax.bar(labels, counts, color=["#34699A", "#EF9B3A", "#B64856"][:len(counts)])
        ax.bar_label(bars, labels=[f"{count:,}\n({count / sum(counts):.1%})" for count in counts], padding=4)
        ax.set_title(title)
        ax.set_ylabel("Số bản ghi")
        ax.set_ylim(0, max(counts) * 1.25)
    save_figure("eda_class_distribution", fig)

    missing = schema.loc[schema.missing_percent > 0].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(missing.feature, missing.missing_percent, color="#34699A")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in missing.missing_percent], padding=4, fontsize=9)
    ax.set_xlim(0, 108)
    ax.set(xlabel="Tỷ lệ thiếu (%)", title="Dữ liệu thiếu sau chuẩn hóa '?' / Unknown/Invalid")
    save_figure("eda_missing_values", fig)

    engineered = preprocessor.named_steps["clinical_features"].transform(X_train)
    report = get_numeric_transform_report(preprocessor, X_train)
    report.to_csv(tables / "training_numeric_transforms.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, feature in zip(axes.flat, NUMERIC_COLUMNS):
        values = pd.to_numeric(X_train[feature], errors="coerce").dropna()
        sns.histplot(values, bins=min(30, max(2, values.nunique())), ax=ax, color="#34699A")
        ax.set_title(f"{feature}\nskew Train = {values.skew():.2f}", fontsize=10)
        ax.set_ylabel("Số bản ghi")
    fig.suptitle("Phân phối biến số trên Train trước biến đổi", y=1.02)
    save_figure("eda_train_numeric_distributions", fig)

    positions = np.arange(len(report))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions - 0.2, report.skew_before, width=0.4, label="Trước log1p")
    ax.bar(positions + 0.2, report.skew_after, width=0.4, label="Sau quy tắc log1p")
    ax.set_xticks(positions, report.feature, rotation=60, ha="right", fontsize=9)
    ax.axhline(preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["log"].skew_threshold,
               color="#B64856", linestyle="--", label="Ngưỡng skewness")
    ax.set(title="Quyết định biến đổi học trên Train; sau đó dùng RobustScaler", ylabel="Skewness")
    ax.legend()
    save_figure("eda_skewness_before_after", fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    for ax, diagnosis in zip(axes, DIAG_COLUMNS):
        counts = engineered[f"{diagnosis}_group"].fillna("Missing").value_counts().sort_values()
        ax.barh(counts.index, counts.values, color="#34699A")
        ax.set(title=f"{diagnosis}: nhóm ICD-9 trên Train", xlabel="Số bản ghi")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    save_figure("eda_train_diagnosis_groups", fig)
    return paths
