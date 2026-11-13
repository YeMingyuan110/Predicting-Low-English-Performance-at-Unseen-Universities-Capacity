"""无泄漏数据审计、确定性清洗和固定高校外层折。

本文件只执行不依赖测试结果的确定性操作。插补、标准化、类别词表、低表现
阈值和校准器必须在训练折内部拟合，不能在本阶段对全样本拟合。论文候选表
写入本模块 results；模型间共享的最小过程文件写入项目 artifacts。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.json"
DEFAULT_RESULTS = MODULE_DIR / "results"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"
SAMPLE_ID = "sample_id"
INSTITUTION = "institution_id"
OUTER_FOLD = "outer_fold"
MISSING_TOKEN = "__MISSING__"

EXPECTED_COLUMNS = [
    "性别", "父亲受教育程度", "母亲受教育程度", "父亲职业", "母亲职业",
    "家庭社会经济阶层", "SISBEN社会福利等级", "家庭人口数", "家庭互联网",
    "电视", "电脑", "洗衣机", "微波炉", "汽车", "DVD设备", "冰箱", "固定电话",
    "手机", "家庭收入", "学生就业情况", "高中学校名称", "高中学校性质",
    "高中学校类型", "高中数学成绩", "高中批判性阅读成绩", "高中公民素养成绩",
    "高中生物成绩", "高中英语成绩", "高校名称", "工程专业", "大学定量推理成绩",
    "大学批判性阅读成绩", "大学公民素养成绩", "大学英语成绩", "大学书面表达成绩",
    "工程项目规划成绩", "大学综合总分", "百分位排名", "双十分位等级", "四分位等级",
    "学生社会经济水平", "高校社会经济水平",
]

T0_NUMERIC = [
    "高中数学成绩", "高中批判性阅读成绩", "高中公民素养成绩",
    "高中生物成绩", "高中英语成绩",
]
DIGITAL_COLUMNS = ["家庭互联网", "电脑", "手机"]
T2_NUMERIC = [*T0_NUMERIC, "DAI"]
T2_CATEGORICAL = [
    "高中学校性质", "高中学校类型", "工程专业", "父亲受教育程度", "母亲受教育程度",
    "父亲职业", "母亲职业", "家庭社会经济阶层", "SISBEN社会福利等级", "家庭收入",
    "家庭人口数", "学生就业情况", "电视", "洗衣机", "微波炉", "汽车", "DVD设备",
    "冰箱", "固定电话",
]
TARGET = "大学英语成绩"
AUDIT_COLUMNS = ["性别", "学生社会经济水平", "高校社会经济水平"]

if len(T2_NUMERIC) + len(T2_CATEGORICAL) != 25:
    raise RuntimeError("T2协议必须恰好包含25个输入字段")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_institution_name(value: object) -> str:
    """保守规范空白和大小写；不自动合并可能不同的校区或法人。"""
    if pd.isna(value):
        raise ValueError("高校名称存在缺失，无法执行高校隔离验证")
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        raise ValueError("高校名称存在缺失，无法执行高校隔离验证")
    return text.casefold()


def validate_schema(columns: Iterable[str]) -> None:
    actual = [str(column).strip() for column in columns]
    missing = [column for column in EXPECTED_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in EXPECTED_COLUMNS]
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            "工作簿字段契约不匹配，拒绝按列位置静默读取。"
            f" missing={missing}, extra={extra}, order_changed={not missing and not extra}"
        )


def load_raw_dataset(workbook: Path, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet_name, engine="openpyxl")
    frame.columns = [str(column).strip() for column in frame.columns]
    validate_schema(frame.columns)
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def add_dai(frame: pd.DataFrame) -> pd.DataFrame:
    """构造透明的0--3设备可及性计数；它不是数字素养量表。"""
    result = frame.copy()
    mapping = {"是": 1.0, "否": 0.0}
    pieces: list[np.ndarray] = []
    invalid: dict[str, list[str]] = {}
    for column in DIGITAL_COLUMNS:
        values = result[column].astype("string").str.strip()
        unknown = sorted(set(values.dropna().unique()).difference(mapping))
        if unknown:
            invalid[column] = unknown
        pieces.append(values.map(mapping).to_numpy(dtype=float))
    if invalid:
        raise ValueError(f"DAI字段存在未定义取值: {invalid}")
    matrix = np.column_stack(pieces)
    result["DAI"] = np.nansum(matrix, axis=1)
    # 任一设备字段缺失时，不能把缺失默认为“没有设备”。
    result.loc[np.isnan(matrix).any(axis=1), "DAI"] = np.nan
    return result


def clean_category(series: pd.Series, zero_is_missing: bool) -> pd.Series:
    values = series.astype("string").str.strip().replace({"": pd.NA})
    if zero_is_missing:
        values = values.replace({"0": pd.NA})
    return values.fillna(MISSING_TOKEN)


def build_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """生成未拟合全局统计量的分析表。"""
    derived = add_dai(raw)
    result = pd.DataFrame(index=raw.index)
    result[SAMPLE_ID] = np.arange(1, len(raw) + 1, dtype=np.int64)
    # 跨模块使用的稳定匿名行键；不包含姓名、学号等直接身份信息。
    result["row_id"] = result[SAMPLE_ID]
    result[INSTITUTION] = raw["高校名称"].map(normalize_institution_name)
    result["institution_label_original"] = raw["高校名称"].astype("string")
    for column in T2_NUMERIC:
        result[column] = pd.to_numeric(derived[column], errors="raise")
    household_missing = {
        "父亲受教育程度", "母亲受教育程度", "父亲职业", "母亲职业", "家庭社会经济阶层",
        "SISBEN社会福利等级", "家庭收入", "家庭人口数", "学生就业情况",
    }
    for column in T2_CATEGORICAL:
        result[column] = clean_category(derived[column], column in household_missing)
    for column in AUDIT_COLUMNS:
        result[column] = clean_category(raw[column], zero_is_missing=False)
    result[TARGET] = pd.to_numeric(raw[TARGET], errors="raise")
    if not np.isfinite(result[TARGET].to_numpy(float)).all():
        raise ValueError("英语结果变量存在缺失或非有限值；结果变量禁止插补")
    if np.isinf(result[T2_NUMERIC].to_numpy(float)).any():
        raise ValueError("预测变量存在无穷值；请核查来源，禁止静默替换")
    if result[SAMPLE_ID].duplicated().any():
        raise RuntimeError("样本编号不唯一")
    return result.reset_index(drop=True)


def assign_outer_folds(frame: pd.DataFrame, n_splits: int) -> pd.Series:
    if n_splits < 2 or frame[INSTITUTION].isna().any():
        raise ValueError("外层折数须至少为2且高校标识不得缺失")
    if frame[INSTITUTION].nunique() < n_splits:
        raise ValueError("高校数量不足以创建指定外层折")
    folds = np.full(len(frame), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, test_index) in enumerate(
        splitter.split(frame, groups=frame[INSTITUTION]), start=1
    ):
        folds[test_index] = fold
    if np.any(folds < 1):
        raise RuntimeError("存在未分配外层折的样本")
    check = pd.DataFrame({INSTITUTION: frame[INSTITUTION], OUTER_FOLD: folds})
    if not check.groupby(INSTITUTION)[OUTER_FOLD].nunique().eq(1).all():
        raise RuntimeError("同一高校被分到多个外层测试折")
    return pd.Series(folds, index=frame.index, name=OUTER_FOLD)


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "T0":
        return list(T0_NUMERIC), []
    if feature_set == "T2":
        return list(T2_NUMERIC), list(T2_CATEGORICAL)
    raise ValueError("feature_set必须为T0或T2")


def make_sklearn_preprocessor(feature_set: str) -> ColumnTransformer:
    numeric, categorical = feature_columns(feature_set)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transformers: list[tuple] = [("numeric", numeric_pipe, numeric)]
    if categorical:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN, keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    # 本数据只有25个原始字段，稠密独热保证HGB等模型与其他基线可共用协议。
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


@dataclass
class TorchFoldPreprocessor:
    """只在当前训练折拟合的ESD输入预处理器。"""

    feature_set: str

    def fit(self, frame: pd.DataFrame) -> "TorchFoldPreprocessor":
        self.numeric_columns, self.categorical_columns = feature_columns(self.feature_set)
        # 固定原始数值token数，避免某个折的缺失指示列改变理论分组索引。
        self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        numeric = self.numeric_imputer.fit_transform(frame[self.numeric_columns])
        self.numeric_scaler = StandardScaler().fit(numeric)
        if self.categorical_columns:
            self.ordinal = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-1, dtype=np.float64,
            ).fit(self._categories(frame))
            self.cardinalities = [len(values) + 1 for values in self.ordinal.categories_]
        else:
            self.ordinal = None
            self.cardinalities = []
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = self.numeric_scaler.transform(
            self.numeric_imputer.transform(frame[self.numeric_columns])
        ).astype(np.float32)
        if self.ordinal is None:
            categorical = np.empty((len(frame), 0), dtype=np.int64)
        else:
            encoded = self.ordinal.transform(self._categories(frame))
            for index, known in enumerate(self.ordinal.categories_):
                encoded[:, index] = np.where(encoded[:, index] < 0, len(known), encoded[:, index])
            categorical = np.rint(encoded).astype(np.int64)
        if not np.isfinite(numeric).all():
            raise ValueError("折内预处理后出现非有限数值")
        return numeric, categorical

    def _categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        """显式缺失类别保留；未见类别统一映射到词表末尾的UNKNOWN索引。"""
        return frame[self.categorical_columns].astype("string").fillna(MISSING_TOKEN).astype(str)


def cohort_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "scope": "overall", "outer_fold": "all", "students": len(frame),
        "institution_labels": frame[INSTITUTION].nunique(),
        "english_mean": frame[TARGET].mean(), "english_sd": frame[TARGET].std(ddof=1),
    }]
    for fold, part in frame.groupby(OUTER_FOLD, sort=True):
        rows.append({
            "scope": "outer_test", "outer_fold": int(fold), "students": len(part),
            "institution_labels": part[INSTITUTION].nunique(),
            "english_mean": part[TARGET].mean(), "english_sd": part[TARGET].std(ddof=1),
        })
    return pd.DataFrame(rows)


def feature_audit_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set in ("T0", "T2"):
        numeric, categorical = feature_columns(feature_set)
        for column in [*numeric, *categorical]:
            rows.append({
                "feature_set": feature_set,
                "feature": column,
                "type": "numeric" if column in numeric else "categorical",
                "missing_n": int(frame[column].isna().sum() + frame[column].eq(MISSING_TOKEN).sum()),
                "unique_n": int(frame[column].nunique(dropna=False)),
                "interpretation": "confirmatory_temporally_eligible" if feature_set == "T0" else "retrospective_upper_bound",
            })
    return pd.DataFrame(rows)


def prepare_project_data(
    workbook: Path | None = None,
    sheet_name: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> tuple[pd.DataFrame, dict]:
    config = load_config(config_path)
    workbook = workbook or Path(config["data_workbook"])
    sheet_name = sheet_name or config["data_sheet"]
    raw = load_raw_dataset(workbook, sheet_name)
    frame = build_analysis_frame(raw)
    frame[OUTER_FOLD] = assign_outer_folds(frame, int(config["outer_folds"]))
    return frame, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--sheet")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    frame, config = prepare_project_data(args.workbook, args.sheet)
    args.results.mkdir(parents=True, exist_ok=True)
    cohort_table(frame).to_csv(
        args.results / "Table_01_Cohort_and_Outer_Fold_Audit.csv", index=False, encoding="utf-8-sig"
    )
    feature_audit_table(frame).to_csv(
        args.results / "Table_02_Feature_Availability_Audit.csv", index=False, encoding="utf-8-sig"
    )

    if not args.no_artifacts:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.artifact_dir / "analysis_frame.csv", index=False, encoding="utf-8-sig")

    workbook = args.workbook or Path(config["data_workbook"])
    manifest = {
        "protocol": "deterministic cleaning followed by university-disjoint five-fold assignment",
        "input_workbook": str(workbook),
        "input_sha256": sha256(workbook),
        "students": len(frame),
        "institution_labels": int(frame[INSTITUTION].nunique()),
        "fold_institution_counts": frame.groupby(OUTER_FOLD)[INSTITUTION].nunique().to_dict(),
        "global_transform_fit": False,
        "python": sys.version,
    }
    (args.results / "Manifest_01_Data_Preprocessing.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(cohort_table(frame).to_string(index=False))


if __name__ == "__main__":
    main()
