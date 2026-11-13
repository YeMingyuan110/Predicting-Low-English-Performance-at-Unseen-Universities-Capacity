"""汇总第15、25、35百分位低表现定义下的预设阈值敏感性结果。

此脚本不会训练模型。它只读取已经独立完成的q=0.15、q=0.25与q=0.35
结果，从而避免在不同阈值之间混合标签、校准器或容量决策。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def tag(feature_set: str, quantile: float) -> str:
    """与run_pipeline.py生成的目录标签保持一致。"""
    return f"{feature_set}_q{quantile:g}"


def require_one(table: pd.DataFrame, where: pd.Series, description: str) -> pd.Series:
    result = table.loc[where]
    if len(result) != 1:
        raise ValueError(f"{description}应恰有一行，实际为{len(result)}行；请核查模型、路线、校准器和输出目录。")
    return result.iloc[0]


def load_one_quantile(project: Path, quantile: float, feature_set: str, model: str,
                      route: str, calibrator: str, capacity: float, policy: str) -> dict:
    run_tag = tag(feature_set, quantile)
    calibration_path = (project / "03_probability_calibration" / "results" / run_tag /
                        "Table_04_Risk_Discrimination_and_Calibration.csv")
    allocation_path = (project / "05_resource_allocation" / "results" / run_tag /
                       f"{model}_{route}" / "Table_06_Capacity_Allocation_Policies.csv")
    predictions_path = project / "artifacts" / run_tag / "calibrated_outer_oof_predictions.csv"
    for path in (calibration_path, allocation_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(
                f"缺少q={quantile:g}的完整敏感性结果：{path}\n"
                "请先在相同高校隔离、模型与校准协议下完成该分位数的第2—5步。"
            )

    calibration = pd.read_csv(calibration_path)
    c = require_one(
        calibration,
        (calibration["model"] == model) & (calibration["feature_set"] == feature_set) &
        (calibration["route"] == route) & (calibration["calibrator"] == calibrator) &
        (calibration["scope"] == "overall"),
        f"q={quantile:g}的风险评价",
    )
    allocation = pd.read_csv(allocation_path)
    a = require_one(
        allocation,
        np.isclose(allocation["capacity"].to_numpy(float), capacity) &
        (allocation["policy"] == policy),
        f"q={quantile:g}、容量={capacity:g}的名单评价",
    )
    pred = pd.read_csv(predictions_path)
    d = pred[(pred["model"] == model) & (pred["feature_set"] == feature_set) &
             (pred["route"] == route) & (pred["calibrator"] == calibrator)].copy()
    if d.empty or d.row_id.duplicated().any():
        raise ValueError(f"q={quantile:g}的冻结OOF预测为空或学生重复。")

    return {
        "low_performance_quantile": quantile,
        "model": model,
        "feature_set": feature_set,
        "route": route,
        "calibrator": calibrator,
        "n": int(c["n"]),
        "events": int(c["events"]),
        "event_prevalence": float(c["event_prevalence"]),
        "mean_training_threshold": float(d["risk_threshold"].mean()),
        "min_training_threshold": float(d["risk_threshold"].min()),
        "max_training_threshold": float(d["risk_threshold"].max()),
        "AP_PR_AUC": float(c["PR_AUC"]),
        "AP_Skill_Score": float(c["AP_Skill_Score"]),
        "ROC_AUC": float(c["ROC_AUC"]),
        "Recall_at_capacity": float(a["Recall_at_K"]),
        "capacity": float(a["capacity"]),
        "policy": policy,
    }


def run(args: argparse.Namespace) -> None:
    if sorted(args.quantiles) != [0.15, 0.25, 0.35]:
        raise ValueError("预设阈值敏感性分析必须同时使用0.15、0.25和0.35。")
    if not 0 < args.capacity <= 1:
        raise ValueError("capacity必须位于(0,1]。")
    project = Path(args.project_root)
    rows = [load_one_quantile(project, q, args.feature_set, args.model, args.route,
                              args.calibrator, args.capacity, args.policy)
            for q in sorted(args.quantiles)]
    result = pd.DataFrame(rows)
    # 各阈值分析必须评价同一批学生，才能把差异解释为标签定义的敏感性。
    if result["n"].nunique() != 1:
        raise ValueError("不同分位数的OOF学生数不一致；不能比较。")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "Table_08_Low_Performance_Threshold_Sensitivity.csv",
                  index=False, encoding="utf-8-sig")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--quantiles", nargs="+", type=float, default=[0.15, 0.25, 0.35])
    p.add_argument("--feature-set", choices=["T0", "T2"], default="T0")
    p.add_argument("--model", default="ESD_MTMS_Net_M1")
    p.add_argument("--route", choices=["S", "D", "J"], default="J")
    p.add_argument("--calibrator", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--capacity", type=float, default=0.10)
    p.add_argument("--policy", default="P1_global_top_k",
                   choices=["P1_global_top_k", "P2_proportional_school_quota"])
    p.add_argument("--output", default=str(HERE / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(cli())
