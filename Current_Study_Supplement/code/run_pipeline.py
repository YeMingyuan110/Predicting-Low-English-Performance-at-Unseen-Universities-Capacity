"""按冻结协议顺序运行研究模块。

本入口不替代各模块的参数说明。正式全量运行前应先逐模块检查输出；默认只
串联已有脚本，不自动忽略失败，也不会在某个模型表现不佳时改动协议。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    ROOT / "01_data_preprocessing" / "data_preprocessing.py",
    ROOT / "02_model_prediction" / "validation_and_tuning.py",
    ROOT / "03_probability_calibration" / "probability_calibration.py",
    ROOT / "04_social_fairness" / "social_fairness.py",
    ROOT / "05_resource_allocation" / "resource_allocation.py",
    ROOT / "06_review_triage" / "review_triage.py",
]


def main() -> None:
    cfg=json.loads((ROOT/"config/experiment_config.json").read_text(encoding="utf-8-sig"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--stop", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--feature-set", choices=("T0", "T2"), default="T0")
    parser.add_argument("--risk-quantile",type=float,default=cfg["low_performance_quantile"])
    parser.add_argument("--models",default=",".join(cfg["primary_baseline_models"]+cfg["supplementary_models"]))
    parser.add_argument("--training-backend",default="torch",choices=["torch","lightning"])
    parser.add_argument("--checkpoint-dir",default=None,
                        help="已核验的中断训练检查点目录；仅在从第2步恢复同一训练时传入。")
    parser.add_argument("--audit-model",default="ESD_MTMS_Net_M1")
    parser.add_argument("--audit-route",default="J",choices=["S","D","J"])
    parser.add_argument("--audit-calibrator", default="nested_selected",
                        help="下游审计唯一概率口径；嵌套选择协议默认nested_selected。")
    parser.add_argument("--calibration-version", default="nested_calibration_selection",
                        help="校准及其下游输出的子目录名称，避免覆盖历史Platt结果。")
    parser.add_argument("--bootstrap-workers",type=int,default=8,
                        help="容量阶段并行高校聚类重抽样的线程数；不改变抽样方案。")
    args = parser.parse_args()
    if args.start > args.stop:
        raise ValueError("--start 不能大于 --stop")
    if not 0<args.risk_quantile<1: raise ValueError("风险分位数必须在0至1之间")
    tag=f"{args.feature_set}_q{args.risk_quantile:g}"
    artifacts=ROOT/"artifacts"/tag
    analysis=ROOT/"artifacts/analysis_frame.csv"
    calibration_artifacts=artifacts/args.calibration_version

    for number, script in enumerate(STEPS, start=1):
        if not args.start <= number <= args.stop:
            continue
        command = [sys.executable, str(script)]
        if number == 2:
            command.extend(["--feature-set", args.feature_set,"--risk-quantile",str(args.risk_quantile),
                "--models",args.models,"--training-backend",args.training_backend,
                "--analysis-frame",str(analysis),"--artifact-dir",str(artifacts)])
            if args.checkpoint_dir:
                command.extend(["--checkpoint-dir",args.checkpoint_dir])
        if number==3:
            command.extend(["--raw",str(artifacts/"raw_outer_oof_predictions.csv"),
                "--inner",str(artifacts/"calibration_inner_oof_predictions.csv"),"--artifact-dir",str(calibration_artifacts)])
        if number>=4:
            command.extend(["--predictions",str(calibration_artifacts/"calibrated_outer_oof_predictions.csv"),
                "--feature-set",args.feature_set,"--model",args.audit_model,"--route",args.audit_route,
                "--calibrator",args.audit_calibrator])
            if number in (4,5): command.extend(["--analysis-frame",str(analysis)])
            if number==5:
                # 决策名单与其概率版本必须同源，避免旧Platt与新嵌套选择结果混用。
                command.extend(["--artifact-dir",str(calibration_artifacts),
                    "--bootstrap-workers",str(args.bootstrap_workers)])
            if number in (4,5): command.extend(["--bootstrap",str(cfg["bootstrap_repetitions"])])
            if number==4:
                command.extend(["--capacity",str(cfg["primary_support_capacity"]),
                    "--min-n",str(cfg["minimum_group_size"]),"--min-events",str(cfg["minimum_group_events"])])
            if number==5:
                command.extend(["--primary-capacity",str(cfg["primary_support_capacity"]),
                    "--capacities",*[str(x) for x in cfg["support_capacities"]]])
            if number==6:
                command.extend(["--support-capacity",str(cfg["primary_support_capacity"]),
                    "--review-budgets",*[str(x) for x in cfg["review_capacities"]]])
        if number>=2:
            destination=script.parent/"results"/tag
            if number==3: destination=destination/args.calibration_version
            if number>=4: destination=destination/f"{args.audit_model}_{args.audit_route}_{args.audit_calibrator}"
            command.extend(["--output",str(destination)])
        print(f"\n[{number}/6] Running {script.name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
