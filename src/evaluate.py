"""Chọn ngưỡng trên validation và đánh giá Test với lựa chọn đã khóa."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}


def validate_scores(y_true, scores):
    y = np.asarray(y_true)
    probabilities = np.asarray(scores, dtype=float)
    if y.ndim != 1 or probabilities.shape != y.shape or not len(y):
        raise ValueError("Nhãn và score phải là hai vector khớp nhau, không rỗng.")
    if not set(np.unique(y)).issubset({0, 1}) or np.unique(y).size != 2:
        raise ValueError("Đánh giá yêu cầu cả hai lớp 0/1.")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Score predict_proba phải hữu hạn và nằm trong [0, 1].")
    return y.astype(int), probabilities


def classification_metrics(y_true, scores, threshold=0.5) -> dict:
    y, probabilities = validate_scores(y_true, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold phải nằm trong [0, 1].")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return {
        "threshold": float(threshold),
        "Recall": float(recall_score(y, predictions, zero_division=0)),
        "Precision": float(precision_score(y, predictions, zero_division=0)),
        "F1": float(f1_score(y, predictions, zero_division=0)),
        "F2": float(fbeta_score(y, predictions, beta=2, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, probabilities)),
        "PR_AUC": float(auc(recall, precision)),
        "Average_Precision": float(average_precision_score(y, probabilities)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "predicted_positive": int(predictions.sum()), "positive_prevalence": float(y.mean()),
    }


def tune_decision_threshold(y_validation, validation_scores, min_precision=0.20, thresholds=None):
    """Max F2 trong 0.10–0.60, ưu tiên tập đạt sàn Precision trên validation.

    Không đạt sàn ở bất kỳ ngưỡng nào: trả best F2 không ràng buộc và gắn
    precision_constraint_met=False, thay vì báo sai là đã đạt yêu cầu.
    """
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision phải trong (0, 1].")
    y, probabilities = validate_scores(y_validation, validation_scores)
    grid = np.round(np.arange(0.10, 0.601, 0.01), 2) if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (grid < .1).any() or (grid > .6).any():
        raise ValueError("Grid threshold phải hữu hạn, không rỗng, trong [0.1, 0.6].")
    rows = []
    for threshold in np.unique(grid):
        predictions = probabilities >= threshold
        rows.append({
            "threshold": float(threshold), "Recall": recall_score(y, predictions, zero_division=0),
            "Precision": precision_score(y, predictions, zero_division=0),
            "F2": fbeta_score(y, predictions, beta=2, zero_division=0),
            "predicted_positive": int(predictions.sum()),
        })
    curve = pd.DataFrame(rows)
    curve["precision_constraint_met"] = (curve.Precision >= min_precision) & (curve.predicted_positive > 0)
    eligible = curve.loc[curve.precision_constraint_met]
    candidates = eligible if len(eligible) else curve
    chosen = candidates.sort_values(["F2", "Recall", "Precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    threshold = float(chosen.threshold)
    result = classification_metrics(y, probabilities, threshold)
    result.update({
        "precision_constraint_met": bool(len(eligible)), "minimum_precision": float(min_precision),
        "selection_partition": "internal_validation_within_outer_train",
    })
    return threshold, curve, result


def short_feature(name, width=48, id_mappings=None):
    label = str(name).split("__", 1)[-1]
    # Giữ ID mã hành chính trong tên rút gọn để các mô tả dài không trùng nhãn.
    for column, prefix in (("discharge_disposition_id", "discharge"),
                            ("admission_source_id", "admission source"),
                            ("admission_type_id", "admission type")):
        if label.startswith(column + "_") and id_mappings is not None:
            description = label[len(column) + 1:]
            code = next((key for key, value in id_mappings[column].items() if value == description), "Missing")
            label = f"{prefix}={code}: {description}"
            break
    return textwrap.shorten(label.replace("_", " "), width=width, placeholder="…")


def create_evaluation_artifacts(models, training_summary, X_test, y_test, X_train, output_dir):
    """Test chỉ được đọc sau khi toàn bộ tham số, threshold và mô hình đã khóa."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import sparse
    import shap

    output_dir = Path(output_dir)
    figures, tables = output_dir / "figures", output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans", palette="colorblind")
    figure_paths = {}

    def save(name, fig):
        fig.tight_layout()
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths[name] = str(path.resolve())

    records = {record["model"]: record for record in training_summary["models"]}
    id_mappings = next(iter(models.values())).named_steps["preprocessing"].named_steps["clinical_features"].id_mappings
    # Một vector score mỗi mô hình cho cả hai operating point.
    scores_by_model = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
    predictions = pd.DataFrame({"y_true": np.asarray(y_test)})
    rows = []
    for name, scores in scores_by_model.items():
        predictions[f"{name}_score"] = scores
        for point, threshold in (("default", .5), ("tuned", records[name]["threshold"])):
            metrics = classification_metrics(y_test, scores, threshold)
            rows.append({"model": name, "label": MODEL_LABELS[name], "operating_point": point,
                         "selected_features": records[name]["selected_features"], **metrics})
            predictions[f"{name}_{point}"] = (scores >= threshold).astype(int)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "model_comparison.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(tables / "test_predictions.csv", index=False)

    # Bảng 6 hàng được minh họa bằng 4 chỉ số có threshold và 2 AUC độc lập threshold.
    names = list(MODEL_LABELS)
    labels = [MODEL_LABELS[name].replace(" ", "\n") for name in names]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    x = np.arange(len(names))
    for ax, metric in zip(axes.flat, ["Recall", "Precision", "F1", "F2", "ROC_AUC", "PR_AUC"]):
        if metric in ("ROC_AUC", "PR_AUC"):
            values = comparison.loc[comparison.operating_point.eq("tuned")].set_index("model").loc[names, metric]
            bars = ax.bar(x, values, width=.55, color="#34699A")
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
            ax.set_title(f"{metric} — không phụ thuộc ngưỡng")
        else:
            for offset, point, label, color in ((-.19, "default", "Ngưỡng 0,50", "#34699A"),
                                                (.19, "tuned", "Ngưỡng chọn trên validation", "#E99B35")):
                values = comparison.loc[comparison.operating_point.eq(point)].set_index("model").loc[names, metric]
                bars = ax.bar(x + offset, values, width=.38, label=label, color=color)
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set_title(metric)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Điểm trên Test")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    save("model_comparison", fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for column, name in enumerate(names):
        for row, point in enumerate(("default", "tuned")):
            result = comparison.loc[comparison.model.eq(name) & comparison.operating_point.eq(point)].iloc[0]
            matrix = np.array([[result.TN, result.FP], [result.FN, result.TP]], dtype=int)
            ax = axes[row, column]
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["0", "1 (<30)"], yticklabels=["0", "1 (<30)"], ax=ax)
            ax.set(title=f"{MODEL_LABELS[name]} | {point} | t={result.threshold:.2f}",
                   xlabel="Dự đoán", ylabel="Thực tế")
    save("confusion_matrices", fig)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(9, 6))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        result = comparison.loc[comparison.model.eq(name) & comparison.operating_point.eq("tuned")].iloc[0]
        ax_roc.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (AUC={result.ROC_AUC:.3f})")
        ax_pr.plot(recall, precision, label=f"{MODEL_LABELS[name]} (PR-AUC={result.PR_AUC:.3f}; AP={result.Average_Precision:.3f})")
        ax_pr.scatter(result.Recall, result.Precision, s=45, zorder=4)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=.5, label="Ngẫu nhiên")
    ax_roc.set(xlabel="False Positive Rate", ylabel="Recall (Sensitivity)", title="ROC: ba thuật toán trên Test")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_pr.axhline(np.mean(y_test), linestyle="--", color="gray", label=f"Baseline tỷ lệ dương={np.mean(y_test):.3f}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall trên Test; điểm tròn = ngưỡng đã khóa",
              xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.legend(fontsize=9)
    save("roc_comparison", fig_roc)
    save("precision_recall_comparison", fig_pr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, names):
        curve = pd.read_csv(tables / f"threshold_curve_{name}.csv")
        for metric in ("Recall", "Precision", "F2"):
            ax.plot(curve.threshold, curve[metric], label=metric)
        ax.axhline(training_summary["config"]["min_precision"], color="gray", linestyle=":", label="Sàn Precision")
        ax.axvline(records[name]["threshold"], color="#B64856", linestyle="--", label=f"Chọn t={records[name]['threshold']:.2f}")
        constraint_text = "đạt sàn Precision" if records[name]["validation_tuned"]["precision_constraint_met"] else "chưa đạt sàn Precision"
        ax.set(title=f"{MODEL_LABELS[name]}\n{constraint_text}", xlabel="Ngưỡng score trên validation", ylabel="Điểm", ylim=(0, 1.03))
        ax.legend(fontsize=8)
    save("threshold_tuning_validation", fig)

    # Hai phương pháp chọn biến được hiển thị từ mô hình chọn trên validation.
    chosen_name = training_summary["chosen_model"]
    selection = pd.read_csv(tables / f"feature_selection_{chosen_name}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    for ax, score_column, title in ((axes[0], "anova_f", "ANOVA F-test"),
                                    (axes[1], "lightgbm_gain", "LightGBM gain")):
        top = selection.nlargest(20, score_column).sort_values(score_column)
        positions = np.arange(len(top))
        ax.barh(positions, top[score_column], color="#34699A")
        ax.set_yticks(positions, [short_feature(name, 43, id_mappings) for name in top.feature])
        ax.set(title=f"Top 20 theo {title} (fit trên Train)", xlabel=title)
        ax.tick_params(axis="y", labelsize=9)
    save("feature_selection_methods", fig)

    for name, model in models.items():
        selector, classifier = model.named_steps["feature_selection"], model.named_steps["classifier"]
        selected_names = selector.get_feature_names_out(model.named_steps["preprocessing"].get_feature_names_out())
        values = classifier.coef_.ravel() if name == "logistic_regression" else classifier.feature_importances_
        importance = pd.DataFrame({"feature": selected_names, "importance": np.abs(values), "signed_value": values})
        importance = importance.sort_values("importance", ascending=False)
        importance.to_csv(tables / f"model_importance_{name}.csv", index=False)
        top = importance.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 8))
        positions = np.arange(len(top))
        ax.barh(positions, top.importance, color="#34699A")
        ax.set_yticks(positions, [short_feature(feature, 58, id_mappings) for feature in top.feature])
        kind = "|hệ số| trên biến đã biến đổi" if name == "logistic_regression" else ("MDI impurity importance" if name == "random_forest" else "Gain importance")
        ax.set(title=f"{MODEL_LABELS[name]}: Top 20 đặc trưng trên Train", xlabel=kind)
        ax.tick_params(axis="y", labelsize=9)
        save(f"feature_importance_{name}", fig)

    # SHAP trên mẫu TRAIN, không dùng Test để tìm thêm biến/tối ưu mô hình.
    lgbm = models["lightgbm"]
    sample = X_train.sample(n=min(training_summary["config"]["shap_samples"], len(X_train)),
                            random_state=training_summary["config"]["random_state"])
    transformed = lgbm.named_steps["preprocessing"].transform(sample)
    selected = lgbm.named_steps["feature_selection"].transform(transformed)
    dense = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    feature_names = lgbm.named_steps["feature_selection"].get_feature_names_out(
        lgbm.named_steps["preprocessing"].get_feature_names_out())
    explainer = shap.TreeExplainer(lgbm.named_steps["classifier"], model_output="raw", feature_perturbation="tree_path_dependent")
    explanation = explainer(dense)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    raw_scores = lgbm.named_steps["classifier"].booster_.predict(dense, raw_score=True)
    shap_raw_scores = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
    np.testing.assert_allclose(shap_raw_scores, raw_scores, atol=1e-5, rtol=1e-5)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(explanation.values).mean(axis=0)})\
        .sort_values("mean_abs_shap", ascending=False).to_csv(tables / "shap_lightgbm_importance.csv", index=False)
    # Ma trận SHAP lưu tên đầy đủ; tên trên hình rút ngắn để dễ đọc.
    np.savez_compressed(output_dir / "artifacts" / "shap_lightgbm.npz",
                        values=explanation.values, base_values=explanation.base_values,
                        feature_names=np.asarray(feature_names, dtype=str), train_row_index=sample.index.to_numpy())
    explanation.feature_names = [short_feature(name, 52, id_mappings) for name in feature_names]
    shap.plots.beeswarm(explanation, max_display=20, show=False, plot_size=(12, 9))
    plt.gca().set_title("LightGBM SHAP trên mẫu Train (raw log-odds)", pad=15)
    save("shap_lightgbm_beeswarm", plt.gcf())
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.gcf().set_size_inches(12, 9)
    plt.gca().set_title("LightGBM: trung bình |SHAP| trên mẫu Train", pad=15)
    save("shap_lightgbm_bar", plt.gcf())
    return comparison, figure_paths


def write_experiment_report(training_summary, comparison, figure_paths, cohort_audit, output_dir):
    """Báo cáo Markdown dùng trực tiếp cho việc viết báo cáo môn học."""
    output_dir = Path(output_dir)
    chosen = training_summary["chosen_model"]
    record = next(row for row in training_summary["models"] if row["model"] == chosen)
    lines = [
        "# Kết quả dự đoán tái nhập viện <30 ngày", "",
        f"Cohort: {cohort_audit['cohort_rows']:,} bệnh nhân; tỷ lệ dương {cohort_audit['positive_rate']:.2%}.",
        f"Development: {training_summary['development_rows']:,}; validation chọn ngưỡng: {training_summary['threshold_validation_rows']:,}; "
        f"outer Train: {training_summary['outer_train_rows']:,}; Test: {int(comparison.iloc[0][['TN','FP','FN','TP']].sum()):,}.", "",
        "## Quy trình và lựa chọn đã khóa trước Test", "",
        f"RandomizedSearchCV: {training_summary['config']['n_iter']} cấu hình mỗi thuật toán, "
        f"{training_summary['config']['cv_folds']}-fold stratified CV, refit theo Average Precision trên development.",
        "Pipeline tiền xử lý + ANOVA/LightGBM selector đều fit riêng trong mỗi fold; số cột k là siêu tham số CV.",
        "Dùng trọng số lớp; không dùng SMOTE. Ngưỡng quét 0,10–0,60, bước 0,01 trên validation tách biệt.",
        f"Sàn Precision trên validation = {training_summary['config']['min_precision']:.0%}; tối đa F2 trong các ngưỡng đạt sàn. "
        "Nếu không có ngưỡng đạt, dùng best F2 và báo rõ chưa đạt sàn.",
        f"Chọn **{MODEL_LABELS[chosen]}** theo validation; threshold **{record['threshold']:.2f}**. "
        f"Validation Recall={record['validation_tuned']['Recall']:.4f}, Precision={record['validation_tuned']['Precision']:.4f}, "
        f"F2={record['validation_tuned']['F2']:.4f}; đạt sàn={record['validation_tuned']['precision_constraint_met']}.",
        "Sau đó refit cả ba pipeline trên toàn outer Train và đánh giá Test. Không chọn lại mô hình/ngưỡng từ bảng Test.", "",
        "## So sánh trên Test", "",
        "| Thuật toán | Ngưỡng | Recall | Precision | F1 | F2 | ROC-AUC | PR-AUC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"| {row.label} ({row.operating_point}) | {row.threshold:.2f} | " +
                     " | ".join(f"{row[metric]:.4f}" for metric in ["Recall", "Precision", "F1", "F2", "ROC_AUC", "PR_AUC", "Average_Precision"]) + " |")
    lines += ["", "## Ngưỡng đã khóa từ validation", "",
              "| Thuật toán | k | Threshold | Recall val | Precision val | F2 val | Đạt sàn Precision |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for candidate in training_summary["models"]:
        validation = candidate["validation_tuned"]
        status = "Có" if validation["precision_constraint_met"] else "Không — dùng best F2 không ràng buộc"
        lines.append(f"| {candidate['label']} | {candidate['selected_features']} | {candidate['threshold']:.2f} | "
                     f"{validation['Recall']:.4f} | {validation['Precision']:.4f} | {validation['F2']:.4f} | {status} |")
    lines += ["", "Threshold thỏa sàn Precision có thể cao hơn 0,50, làm giảm Recall/F2 so với ngưỡng mặc định. "
              "Đây là tối ưu có ràng buộc Precision, không phải cam kết mọi metric đều tăng sau tuning.", "",
              "PR-AUC dùng tích phân hình thang của đường Precision–Recall; AP dùng average_precision_score, "
              "không nội suy. Hai giá trị được báo riêng. ROC-AUC/PR-AUC/AP không đổi khi chỉ đổi threshold.",
              "Sàn Precision là ràng buộc chọn trên validation, không bảo đảm Precision Test cũng đạt sàn.", "",
              "## Đặc trưng và diễn giải", "",
              "Hai bảng xếp hạng là ANOVA F và LightGBM gain; selector lấy trung bình thứ hạng để chọn top-k.",
              "Các p-value ANOVA là chưa hiệu chỉnh đa kiểm định và chỉ để tham khảo, không dùng để tuyên bố ý nghĩa lâm sàng.",
              "Importance/SHAP cho thấy mô hình dùng biến nào, không chứng minh nguyên nhân tái nhập viện. "
              "SHAP được tính trên mẫu Train của LightGBM theo raw log-odds; đã kiểm tra tính cộng khớp output mô hình.",
              "Các score predict_proba có trọng số lớp chưa được hiệu chỉnh xác suất; không diễn giải trực tiếp là nguy cơ lâm sàng tuyệt đối.",
              "CSV không có timestamp nhập viện: encounter_id nhỏ nhất chỉ là quy ước chọn bản ghi đại diện. "
              "Thời điểm dự đoán là khi xuất viện; kết quả trên dữ liệu lịch sử này chưa phải kiểm định triển khai thực tế.", "",
              "## Biểu đồ", ""]
    for name, path in figure_paths.items():
        lines.extend([f"### {name}", "", f"![{name}](figures/{Path(path).name})", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
