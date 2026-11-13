"""Estimate the unconditional university-level ICC of Saber Pro English scores.

This is a pre-model descriptive analysis.  It does not use predictors, model
outputs, outer-fold labels, or any training result.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep this small descriptive calculation from competing with model training.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INSTITUTION = "institution_id"
TARGET = "大学英语成绩"


def estimate_icc(frame: pd.DataFrame) -> dict:
    """Estimate the unconditional ICC by one-way random-effects moments.

    The estimator uses the standard unbalanced-group effective size n0.  It is
    the dependency-free descriptive counterpart of an intercept-only random-
    intercept model and is appropriate here because ICC is not used for model
    selection or inferential causal claims.
    """
    needed = {INSTITUTION, TARGET}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"分析表缺少字段: {sorted(missing)}")
    data = frame[[INSTITUTION, TARGET]].dropna().rename(
        columns={INSTITUTION: "university", TARGET: "english_score"}
    )
    if data.university.nunique() < 2:
        raise ValueError("计算高校ICC至少需要两个高校分组。")
    grouped = data.groupby("university", observed=True)["english_score"]
    sizes = grouped.size()
    means = grouped.mean()
    n, groups = int(len(data)), int(sizes.size)
    grand_mean = float(data["english_score"].mean())
    ss_between = float(np.sum(sizes.to_numpy(float) * (means.to_numpy(float) - grand_mean) ** 2))
    ss_within = float(grouped.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum())
    ms_between = ss_between / (groups - 1)
    ms_within = ss_within / (n - groups)
    effective_group_size = (n - float(np.sum(sizes.to_numpy(float) ** 2)) / n) / (groups - 1)
    # A negative moment component is a sampling artefact and corresponds to a
    # boundary estimate of zero between-group variance.
    between = max(0.0, (ms_between - ms_within) / effective_group_size)
    within = ms_within
    total = between + within
    sizes = data.groupby("university", observed=True).size()
    return {
        "outcome": "Saber Pro English score",
        "grouping_unit": "university name label",
        "estimator": "one-way random-effects ANOVA moment estimator (unbalanced groups)",
        "students": int(len(data)),
        "university_labels": int(sizes.size),
        "mean_students_per_university": float(sizes.mean()),
        "median_students_per_university": float(sizes.median()),
        "min_students_per_university": int(sizes.min()),
        "max_students_per_university": int(sizes.max()),
        "between_university_variance": between,
        "within_university_variance": within,
        "effective_group_size": effective_group_size,
        "unconditional_icc": between / total if total > 0 else float("nan"),
        "interpretation_boundary": (
            "Descriptive clustering by university label; not a causal university effect and not a model-performance metric."
        ),
    }


def run(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.analysis_frame)
    # The primary estimate retains every labelled university.  The additional
    # rows are a transparent small-group check, because a few labels contain
    # only very small numbers of students.  They are not alternative analyses
    # used for selecting a preferred ICC.
    rows: list[dict] = []
    for minimum_size in args.minimum_group_sizes:
        counts = frame.groupby(INSTITUTION, observed=True)[TARGET].transform("size")
        subset = frame.loc[counts >= minimum_size].copy()
        result = estimate_icc(subset)
        result["minimum_students_per_label_included"] = int(minimum_size)
        result["analysis_role"] = (
            "Primary estimate: all university labels" if minimum_size == 1
            else "Sensitivity check: excludes labels below the stated size"
        )
        rows.append(result)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "Table_01_English_Score_University_ICC_with_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = rows[0]
    print(
        "ICC_OK "
        f"students={result['students']} universities={result['university_labels']} "
        f"icc={result['unconditional_icc']:.6f}"
    )


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-frame", default=str(ROOT / "artifacts" / "analysis_frame.csv"))
    p.add_argument("--output", default=str(HERE / "results"))
    p.add_argument(
        "--minimum-group-sizes",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Primary all-label estimate (1) plus optional small-label sensitivity checks.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(cli())
