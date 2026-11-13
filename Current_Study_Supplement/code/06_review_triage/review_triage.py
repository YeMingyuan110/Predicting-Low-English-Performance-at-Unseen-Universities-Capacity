"""有限复核预算：只评价Error Capture@Q，不推断真实人工干预收益。"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"03_probability_calibration"))
from probability_calibration import select_predictions


def strategy_scores(d,support_capacity):
    p=np.clip(d.probability.to_numpy(),1e-9,1-1e-9)
    k=max(1,int(round(len(d)*support_capacity))); cutoff=np.sort(p)[-k]
    scores={"Random":np.zeros(len(d)),"Probability_entropy":-(p*np.log(p)+(1-p)*np.log(1-p)),
            "TopK_boundary":-np.abs(p-cutoff)}
    if "seed_count" in d and d.seed_count.min()>=2:
        if "seed_disagreement" not in d: raise ValueError("缺少已经计算的种子分歧")
        scores["Seed_disagreement"]=d.seed_disagreement.to_numpy()
    if "institution_input_shift" in d:
        scores["Institution_input_shift"]=d.institution_input_shift.to_numpy()
    return scores


def run(args):
    d=select_predictions(pd.read_csv(args.predictions),args.model,args.route,args.calibrator,args.feature_set)
    if any(not 0<v<=1 for v in [*args.review_budgets,args.support_capacity]):
        raise ValueError("容量必须在(0,1]内")
    d=d.sort_values("row_id",kind="stable").reset_index(drop=True)
    k=max(1,int(round(args.support_capacity*len(d))))
    topk=np.zeros(len(d),dtype=int)
    topk[np.argsort(-d.probability.to_numpy(),kind="stable")[:k]]=1
    # The formal audit concerns the actual capacity-constrained list.  A fixed
    # 0.5 threshold would answer a different question and is therefore not
    # reported as a second, competing error definition.
    errors={"TopK_selection_error":(topk!=d.observed_risk.to_numpy()).astype(int)}
    scores=strategy_scores(d,args.support_capacity); rows=[]
    for error_definition,error in errors.items():
        total=int(error.sum())
        for budget in args.review_budgets:
            q=max(1,int(round(len(d)*budget)))
            for name,score in scores.items():
                if not np.isfinite(score).all(): raise ValueError("复核信号包含非有限值")
                # 随机复核采用无放回均匀抽样的解析期望，不引入额外Monte Carlo误差。
                if name=="Random": caught=total*q/len(d)
                else:
                    idx=np.argsort(-np.asarray(score),kind="stable")[:q]
                    caught=float(error[idx].sum())
                rows.append({"model":args.model,"route":args.route,"feature_set":args.feature_set,
                    "calibrator":args.calibrator,"error_definition":error_definition,
                    "support_capacity":args.support_capacity,"review_budget":budget,"review_n":q,"strategy":name,
                    "errors_captured":caught,
                    "error_capture_rate":caught/total if total else np.nan})
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"Table_07_Review_Triage_Efficiency.csv",index=False,encoding="utf-8-sig")


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",default=str(ROOT/"artifacts/calibrated_outer_oof_predictions.csv"))
    p.add_argument("--model",default="ESD_MTMS_Net_M1"); p.add_argument("--route",default="J",choices=["S","D","J"])
    p.add_argument("--feature-set",default="T0",choices=["T0","T2"]); p.add_argument("--calibrator",default="platt")
    p.add_argument("--review-budgets",nargs="+",type=float,default=[.05,.10,.20])
    p.add_argument("--support-capacity",type=float,default=.10)
    p.add_argument("--output",default=str(HERE/"results")); return p.parse_args()
if __name__=="__main__": run(cli())
