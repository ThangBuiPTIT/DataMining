"""Tiền xử lý dự đoán tái nhập viện <30 ngày, tại thời điểm xuất viện.

Quy tắc xác định cohort độc lập với nhãn. Mọi thống kê (imputation,
skewness, loại cột, vocabulary, scaling) chỉ được học khi Pipeline.fit().
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted


RANDOM_STATE = 42
TARGET = "readmitted_30d"
ID_COLUMNS = ("encounter_id", "patient_nbr")
EXCLUDED_DISCHARGES = (11, 13, 14, 19, 20, 21)
FIXED_DROP_COLUMNS = ("weight", "examide", "citoglipton")
ADMIN_COLUMNS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
DIAG_COLUMNS = ("diag_1", "diag_2", "diag_3")
LAB_COLUMNS = ("max_glu_serum", "A1Cresult")
NUMERIC_COLUMNS = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)
DRUG_COLUMNS = (
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
)
ACTIVE_DRUG_COLUMNS = tuple(c for c in DRUG_COLUMNS if c not in FIXED_DROP_COLUMNS)
BINARY_COLUMNS = (
    "change", "diabetesMed", *(f"{drug}_prescribed" for drug in ACTIVE_DRUG_COLUMNS),
    *(f"{lab}_measured" for lab in LAB_COLUMNS),
)
REQUIRED_COLUMNS = (
    *ID_COLUMNS, "race", "gender", "age", "weight", *ADMIN_COLUMNS,
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", *DIAG_COLUMNS, "number_diagnoses", *LAB_COLUMNS,
    *DRUG_COLUMNS, "change", "diabetesMed", "readmitted",
)
ENGINEERED_NUMERIC_COLUMNS = (
    "total_prior_visits", "n_medications_changed", "n_medications_increased",
    "n_medications_decreased", "n_drugs_prescribed", "n_medications_unknown",
)
AGE_LEVELS = [f"[{start}-{start + 10})" for start in range(0, 100, 10)]
ORDINAL_LEVELS = {
    "age": AGE_LEVELS,
    "max_glu_serum": ["Norm", ">200", ">300"],
    "A1Cresult": ["Norm", ">7", ">8"],
    **{drug: ["Down", "Steady", "Up"] for drug in ACTIVE_DRUG_COLUMNS},
}
UNKNOWN_DESCRIPTIONS = {"", "null", "not available", "not mapped", "unknown/invalid"}


def normalize_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Không thay chuỗi 'None': ở xét nghiệm, nó có nghĩa là chưa đo."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        result[column] = result[column].mask(result[column].isin(["?", "Unknown/Invalid", ""]), np.nan)
    return result


def _require_columns(frame: pd.DataFrame, columns: tuple | list) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")


def load_diabetes_data(path: str | Path) -> pd.DataFrame:
    """Đọc mã ICD/admin dưới dạng chuỗi, giữ None khác với NaN."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    string_columns = (*ID_COLUMNS, *DIAG_COLUMNS, *ADMIN_COLUMNS, *LAB_COLUMNS)
    frame = pd.read_csv(
        path, keep_default_na=False, na_values=["?", "Unknown/Invalid", ""],
        dtype={column: str for column in string_columns}, low_memory=False,
    )
    _require_columns(frame, REQUIRED_COLUMNS)
    return normalize_missing(frame)


def load_id_mappings(path: str | Path) -> dict[str, dict[str, str]]:
    """IDS_mapping.csv chứa ba bảng nối tiếp; không phải một bảng phẳng."""
    mappings: dict[str, dict[str, str]] = {}
    active_section = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            key = row[0].strip()
            if key in ADMIN_COLUMNS:
                if len(row) != 2 or row[1].strip() != "description":
                    raise ValueError(f"Header không hợp lệ trong {path}: {row}")
                active_section = key
                mappings[key] = {}
            elif active_section is not None and len(row) == 2:
                mappings[active_section][key] = row[1].strip()
            else:
                raise ValueError(f"Dòng mapping không hợp lệ: {row}")
    if set(mappings) != set(ADMIN_COLUMNS):
        raise ValueError("IDS_mapping.csv phải chứa đủ ba bảng mã hành chính.")
    if not set(map(str, EXCLUDED_DISCHARGES)).issubset(mappings["discharge_disposition_id"]):
        raise ValueError("Mapping thiếu mã tử vong/hospice cần kiểm tra.")
    return mappings


def prepare_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chọn ID encounter nhỏ nhất mỗi bệnh nhân, rồi loại tử vong/hospice.

    CSV không chứa timestamp; encounter_id là quy ước đại diện, không được
    xem là bằng chứng về lần nhập viện đầu tiên theo thời gian thực tế.
    """
    _require_columns(raw, REQUIRED_COLUMNS)
    frame = normalize_missing(raw)
    if frame.empty:
        raise ValueError("Dữ liệu rỗng.")
    if frame[list(ID_COLUMNS)].isna().any().any():
        raise ValueError("ID bệnh nhân/lượt nhập viện bị thiếu; không thể chống trùng bệnh nhân.")
    for column in ID_COLUMNS:
        frame[column] = frame[column].astype(str)
    encounter_order = pd.to_numeric(frame["encounter_id"], errors="raise")
    if not np.isfinite(encounter_order).all() or (encounter_order % 1 != 0).any():
        raise ValueError("encounter_id phải là số nguyên hữu hạn.")
    if encounter_order.duplicated().any():
        raise ValueError("encounter_id bị trùng; cần kiểm tra dữ liệu nguồn.")
    if not frame["readmitted"].isin(["<30", ">30", "NO"]).all():
        raise ValueError("readmitted có nhãn thiếu/không hợp lệ; không tự động gán thành lớp 0.")
    frame = frame.assign(_encounter_order=encounter_order).sort_values(
        "_encounter_order", kind="mergesort"
    ).drop_duplicates("patient_nbr", keep="first").drop(columns="_encounter_order")
    unique_count = len(frame)
    discharge = pd.to_numeric(frame["discharge_disposition_id"], errors="coerce")
    excluded = discharge.isin(EXCLUDED_DISCHARGES)
    frame = frame.loc[~excluded].copy()
    frame[TARGET] = frame["readmitted"].eq("<30").astype("int8")
    frame = frame.reset_index(drop=True)
    if frame[TARGET].nunique() != 2:
        raise ValueError("Cohort sau lọc phải còn cả hai lớp mục tiêu.")
    audit = {
        "raw_rows": len(raw), "unique_patients_before_exclusions": unique_count,
        "repeated_encounters_removed": len(raw) - unique_count,
        "death_hospice_removed_after_deduplication": int(excluded.sum()),
        "cohort_rows": len(frame), "positive_count": int(frame[TARGET].sum()),
        "positive_rate": float(frame[TARGET].mean()),
        "excluded_discharge_ids": list(EXCLUDED_DISCHARGES),
        "representative_encounter": "minimum_numeric_encounter_id_before_discharge_filter",
        "chronology_limitation": "No admission timestamp in supplied CSV; ID order is a proxy.",
        "prediction_time": "discharge",
    }
    return frame, audit


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.DataFrame
    test_ids: pd.DataFrame


def split_cohort(cohort: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> DataSplits:
    """Split phân tầng sau deduplicate; không bệnh nhân nào ở cả hai tập."""
    _require_columns(cohort, [*ID_COLUMNS, "readmitted", TARGET])
    if cohort["patient_nbr"].isna().any() or cohort["patient_nbr"].duplicated().any():
        raise ValueError("split_cohort yêu cầu một bản ghi cho mỗi bệnh nhân.")
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm trong (0, 1).")
    train_indices, test_indices = train_test_split(
        cohort.index, test_size=test_size, random_state=random_state, stratify=cohort[TARGET],
    )
    train, test = cohort.loc[train_indices], cohort.loc[test_indices]
    forbidden = [*ID_COLUMNS, "readmitted", TARGET]
    result = DataSplits(
        X_train=train.drop(columns=forbidden).copy(), X_test=test.drop(columns=forbidden).copy(),
        y_train=train[TARGET].copy(), y_test=test[TARGET].copy(),
        train_ids=train[list(ID_COLUMNS)].copy(), test_ids=test[list(ID_COLUMNS)].copy(),
    )
    if set(result.train_ids.patient_nbr) & set(result.test_ids.patient_nbr):
        raise RuntimeError("Phát hiện rò rỉ bệnh nhân giữa Train/Test.")
    return result


def group_icd9(value: object) -> object:
    """Nhóm ICD-9 theo Strack et al. (2014), Table 2; thiếu giữ là NaN."""
    if pd.isna(value) or str(value).strip() in {"", "?", "Unknown/Invalid"}:
        return np.nan
    code = str(value).strip().upper()
    if not re.fullmatch(r"\d{1,3}(?:\.\d+)?", code):
        return "Other"  # Bao gồm E/V; không coi đây là mã số bệnh thông thường.
    category = int(float(code))
    if category == 250:
        return "Diabetes"
    for lower, upper, special, label in (
        (390, 459, 785, "Circulatory"), (460, 519, 786, "Respiratory"),
        (520, 579, 787, "Digestive"), (580, 629, 788, "Genitourinary"),
        (800, 999, -1, "Injury"), (710, 739, -1, "Musculoskeletal"),
        (140, 239, -1, "Neoplasms"),
    ):
        if lower <= category <= upper or category == special:
            return label
    return "Other"


class ClinicalFeatureEngineer(TransformerMixin, BaseEstimator):
    """Biến đổi tất định; không học thống kê hay dùng nhãn mục tiêu.

    Thuốc: mã hóa hướng Down < Steady < Up, No được tách bằng prescribed.
    None của xét nghiệm: severity thiếu + cờ measured=0, không gán là Norm.
    """

    def __init__(self, id_mappings: dict | None = None):
        self.id_mappings = id_mappings

    def fit(self, X: pd.DataFrame, y=None):
        forbidden = {"readmitted", TARGET} & set(X.columns)
        if forbidden:
            raise ValueError(f"Phải tách nhãn trước khi fit preprocessor: {sorted(forbidden)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(self._engineer(X).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        return self._engineer(X).loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        core = [c for c in REQUIRED_COLUMNS if c not in (*ID_COLUMNS, "readmitted", *FIXED_DROP_COLUMNS)]
        _require_columns(X, core)
        frame = normalize_missing(X)
        # Chọn schema rõ ràng để không vô tình đưa cột ngoài danh sách vào mô hình.
        frame = frame.loc[:, core].copy()
        for column in NUMERIC_COLUMNS:
            original = frame[column]
            frame[column] = pd.to_numeric(original, errors="coerce").astype(float)
            if (original.notna() & frame[column].isna()).any():
                raise ValueError(f"Cột {column} chứa giá trị không phải số.")
            if np.isinf(frame[column]).any() or (frame[column].dropna() < 0).any():
                raise ValueError(f"Cột {column} phải không âm và hữu hạn.")
        los = frame["time_in_hospital"].dropna()
        if not los.between(1, 14).all():
            raise ValueError("time_in_hospital phải từ 1 đến 14 ngày theo cohort UCI.")
        frame["total_prior_visits"] = frame[
            ["number_outpatient", "number_emergency", "number_inpatient"]
        ].sum(axis=1, min_count=3)
        drugs = frame[list(ACTIVE_DRUG_COLUMNS)]
        valid_states = {"No", "Down", "Steady", "Up"}
        if any(not set(drugs[column].dropna()).issubset(valid_states) for column in drugs):
            raise ValueError("Trạng thái thuốc phải thuộc No/Down/Steady/Up hoặc thiếu.")
        any_known = drugs.notna().any(axis=1)
        frame["n_medications_unknown"] = drugs.isna().sum(axis=1).astype(float)
        for name, states in (
            ("n_medications_changed", ["Down", "Up"]),
            ("n_medications_increased", ["Up"]),
            ("n_medications_decreased", ["Down"]),
            ("n_drugs_prescribed", ["Down", "Steady", "Up"]),
        ):
            frame[name] = drugs.isin(states).sum(axis=1).astype(float).where(any_known, np.nan)
        for drug in ACTIVE_DRUG_COLUMNS:
            frame[f"{drug}_prescribed"] = frame[drug].map(
                {"No": 0.0, "Down": 1.0, "Steady": 1.0, "Up": 1.0}
            )
            frame[drug] = frame[drug].replace({"No": "Steady"})
        for lab in LAB_COLUMNS:
            allowed = {"None", *ORDINAL_LEVELS[lab]}
            if not set(frame[lab].dropna()).issubset(allowed):
                raise ValueError(f"Giá trị không hợp lệ ở {lab}.")
            frame[f"{lab}_measured"] = frame[lab].map(
                {"None": 0.0, **{level: 1.0 for level in ORDINAL_LEVELS[lab]}}
            )
            frame[lab] = frame[lab].mask(frame[lab].eq("None"), np.nan)
        frame["change"] = frame["change"].map({"No": 0.0, "Ch": 1.0})
        frame["diabetesMed"] = frame["diabetesMed"].map({"No": 0.0, "Yes": 1.0})
        for column in DIAG_COLUMNS:
            frame[f"{column}_group"] = frame.pop(column).map(group_icd9)
        for column in ADMIN_COLUMNS:
            codes = frame[column].map(
                lambda value: str(int(float(value))) if pd.notna(value) else np.nan
            )
            if self.id_mappings is not None:
                frame[column] = codes.map(self.id_mappings[column])
                frame[column] = frame[column].map(
                    lambda value: np.nan if isinstance(value, str) and value.lower() in UNKNOWN_DESCRIPTIONS else value
                )
            else:
                # Không có mapping: vẫn xử lý ID dạng nominal, tuyệt đối không scale ID.
                unknown = {
                    "admission_type_id": {"5", "6", "8"},
                    "discharge_disposition_id": {"18", "25", "26"},
                    "admission_source_id": {"9", "15", "17", "20", "21"},
                }
                frame[column] = codes.where(~codes.isin(unknown[column]), np.nan)
        # sklearn cần np.nan, không phải pd.NA trong các nhánh object.
        return frame.where(frame.notna(), np.nan)


class TrainOnlyColumnFilter(TransformerMixin, BaseEstimator):
    """Loại cột quá thiếu/hằng số, với quyết định chỉ dựa trên dữ liệu fit."""

    def __init__(self, missing_threshold: float = 0.95):
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y=None):
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError("missing_threshold phải trong [0, 1].")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.missing_rates_ = X.isna().mean()
        self.dropped_reasons_ = {}
        for column in X:
            # Xét nghiệm chưa đo là trạng thái có ý nghĩa: giữ kết quả + cờ measured.
            if self.missing_rates_[column] > self.missing_threshold and column not in LAB_COLUMNS:
                self.dropped_reasons_[column] = "missing_above_threshold_on_train"
            elif X[column].nunique(dropna=False) <= 1:
                self.dropped_reasons_[column] = "constant_on_train"
        self.kept_columns_ = [column for column in X if column not in self.dropped_reasons_]
        if not self.kept_columns_:
            raise ValueError("Không còn đặc trưng nào sau bước lọc cột trên Train.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "kept_columns_")
        _require_columns(X, self.kept_columns_)
        return X.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kept_columns_")
        return np.asarray(self.kept_columns_, dtype=object)


class SkewAwareLogTransformer(TransformerMixin, BaseEstimator):
    """Học skewness trên Train; log1p cột không âm có skew > ngưỡng."""

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Nhánh số yêu cầu ma trận 2D không âm, hữu hạn sau impute.")
        self.n_features_in_ = array.shape[1]
        self.skewness_ = pd.DataFrame(array).skew().fillna(0).to_numpy()
        self.log_mask_ = self.skewness_ > self.skew_threshold
        return self

    def transform(self, X):
        check_is_fitted(self, "log_mask_")
        array = np.asarray(X, dtype=float).copy()
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Số cột khác lúc fit SkewAwareLogTransformer.")
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Không log-transform giá trị âm/vô hạn.")
        array[:, self.log_mask_] = np.log1p(array[:, self.log_mask_])
        return array

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "log_mask_")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def select_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in (*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS) if column in frame]


def select_ordinal_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ORDINAL_LEVELS if column in frame]


def select_binary_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column in BINARY_COLUMNS]


def select_nominal_columns(frame: pd.DataFrame) -> list[str]:
    assigned = set(select_numeric_columns(frame) + select_ordinal_columns(frame) + select_binary_columns(frame))
    return [column for column in frame if column not in assigned]


class ClinicalOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Dùng thứ tự miền y tế đã khai báo; unknown/missing không thành mức nguy cơ."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.encoder_ = OrdinalEncoder(
            categories=[[*ORDINAL_LEVELS[column], np.nan] for column in X],
            handle_unknown="use_encoded_value", unknown_value=np.nan,
            encoded_missing_value=np.nan, dtype=np.float64,
        ).fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")
        return self.encoder_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return self.feature_names_in_.copy()


def build_preprocessor(
    id_mappings: dict | None = None, skew_threshold: float = 1.0, missing_threshold: float = 0.95,
) -> Pipeline:
    """Trả về Pipeline CHƯA fit, để dùng lại bên trong mỗi CV fold sau này."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", SkewAwareLogTransformer(skew_threshold=skew_threshold)),
        ("scaler", RobustScaler()),
    ])
    ordinal = Pipeline([
        ("encoder", ClinicalOrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True, keep_empty_features=True)),
    ])
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True, keep_empty_features=True)),
    ])
    nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric, select_numeric_columns),
            ("ordinal", ordinal, select_ordinal_columns),
            ("binary", binary, select_binary_columns),
            ("nominal", nominal, select_nominal_columns),
        ],
        remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True,
    )
    return Pipeline([
        ("clinical_features", ClinicalFeatureEngineer(id_mappings=id_mappings)),
        ("column_filter", TrainOnlyColumnFilter(missing_threshold=missing_threshold)),
        ("columns", columns),
    ])


def get_numeric_transform_report(preprocessor: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    """Xuất quyết định log-transform từ pipeline đã fit; không fit lại."""
    engineered = preprocessor.named_steps["clinical_features"].transform(X_train)
    filtered = preprocessor.named_steps["column_filter"].transform(engineered)
    names = select_numeric_columns(filtered)
    if not names:
        return pd.DataFrame(columns=["feature", "skew_before", "skew_after", "log1p"])
    numeric = preprocessor.named_steps["columns"].named_transformers_["numeric"]
    imputed = numeric.named_steps["imputer"].transform(filtered[names])
    logged = numeric.named_steps["log"].transform(imputed)
    return pd.DataFrame({
        "feature": names, "skew_before": numeric.named_steps["log"].skewness_,
        "skew_after": pd.DataFrame(logged).skew().fillna(0).to_numpy(),
        "log1p": numeric.named_steps["log"].log_mask_,
    })
