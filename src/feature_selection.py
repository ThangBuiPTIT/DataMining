"""Chọn đặc trưng trong từng fold bằng ANOVA và importance của LightGBM."""

from __future__ import annotations

import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted


class HybridFeatureSelector(TransformerMixin, BaseEstimator):
    """Xếp hạng trung bình hai phương pháp, giữ k cột tốt nhất.

    Đây là chọn biến dựa trên liên hệ dự đoán, không phải bằng chứng nhân quả.
    Mọi lần fit chỉ nhận nhãn Train của fold; không dùng Test/validation.
    """

    def __init__(self, k=80, random_state=42, n_jobs=2, n_estimators=100):
        self.k = k
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

    def fit(self, X, y):
        matrix = sparse.csr_matrix(X, dtype=np.float64)
        if matrix.shape[0] != len(y) or np.unique(y).size != 2:
            raise ValueError("Selector cần X/y khớp hàng và đủ hai lớp.")
        if self.k != "all" and (not isinstance(self.k, (int, np.integer)) or self.k < 1):
            raise ValueError("k phải là số nguyên dương hoặc 'all'.")
        self.n_features_in_ = matrix.shape[1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")
            scores, pvalues = f_classif(matrix, y)
        self.anova_scores_ = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max)
        self.anova_pvalues_ = np.nan_to_num(pvalues, nan=1.0)
        self.importance_model_ = LGBMClassifier(
            n_estimators=self.n_estimators, num_leaves=15, max_depth=5,
            learning_rate=0.05, min_child_samples=60, reg_lambda=2.0,
            is_unbalance=True, importance_type="gain", random_state=self.random_state,
            n_jobs=self.n_jobs, verbosity=-1, deterministic=True, force_col_wise=True,
        ).fit(matrix, y)
        self.model_importances_ = self.importance_model_.feature_importances_.astype(float)
        self.anova_ranks_ = rankdata(-self.anova_scores_, method="average")
        self.model_ranks_ = rankdata(-self.model_importances_, method="average")
        self.combined_ranks_ = (self.anova_ranks_ + self.model_ranks_) / 2
        self.ranked_indices_ = np.lexsort((np.arange(self.n_features_in_), self.combined_ranks_))
        count = self.n_features_in_ if self.k == "all" else min(int(self.k), self.n_features_in_)
        # Giữ thứ tự cột gốc để names, hệ số, SHAP và ma trận luôn khớp nhau.
        self.selected_indices_ = np.sort(self.ranked_indices_[:count])
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        matrix = sparse.csr_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Schema đầu vào selector khác lúc fit.")
        return matrix[:, self.selected_indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, "selected_indices_")
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_indices_")
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        if len(names) != self.n_features_in_:
            raise ValueError("Số tên đặc trưng không khớp selector.")
        return names[self.selected_indices_]

    def ranking_table(self, feature_names) -> pd.DataFrame:
        check_is_fitted(self, "selected_indices_")
        return pd.DataFrame({
            "feature": feature_names, "anova_f": self.anova_scores_,
            "anova_pvalue_unadjusted": self.anova_pvalues_, "anova_rank": self.anova_ranks_,
            "lightgbm_gain": self.model_importances_, "lightgbm_rank": self.model_ranks_,
            "combined_rank": self.combined_ranks_, "selected": self.get_support(),
        }).sort_values(["combined_rank", "feature"]).reset_index(drop=True)
