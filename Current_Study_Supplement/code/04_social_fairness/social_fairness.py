"""只报告核心公平审计：群体平均预测偏差与固定容量名单TPR差距。"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"03_probability_calibration"))
from probability_calibration import select_predictions
GROUPS={"Gender":"性别","Student_SES":"学生社会经济水平","Institution_SES":"高校社会经济水平"}


def group_statistics(g,w):
    w=np.asarray(w,float); y=g.observed_risk.to_numpy(float); p=g.probability.to_numpy(float)
    n=w.sum(); events=np.dot(w,y)
    if n<=0: return {}
    avg=lambda v:float(np.dot(w,np.asarray(v))/n)
    s=g.selected.to_numpy(float); residual=g.predicted_score-g.observed_score
    finite=np.isfinite(residual).all()
    return {"n":n,"events":events,
        "mean_residual":avg(residual) if finite else np.nan,
        "TPR":np.dot(w,s*y)/events if events else np.nan}


def audit_dimension(data,col,label,min_n,min_events,boot,seed=42):
    """固定符合人数要求的群体；每次重抽高校保持同一群体集合。"""
    data=data.copy(); data[col]=data[col].fillna("MISSING").astype(str)
    rows=[]; eligible=[]
    for level,g in data.groupby(col,sort=True):
        m=group_statistics(g,np.ones(len(g)))
        ok=m["n"]>=min_n and min(m["events"],m["n"]-m["events"])>=min_events
        rows.append({"dimension":label,"group":level,"status":"reportable" if ok else "small_group",
                     "institutions":g.institution.nunique(),**m})
        if ok: eligible.append(level)
    metrics=["mean_residual","TPR"]
    observed=pd.DataFrame(rows).set_index("group")
    schools=data.institution.unique(); rng=np.random.default_rng(seed); draws={m:[] for m in metrics}
    if len(eligible)>1:
        for _ in range(boot):
            counts=pd.Series(rng.choice(schools,len(schools),replace=True)).value_counts()
            w=data.institution.map(counts).fillna(0).to_numpy()
            estimates=[]
            for level in eligible:
                mask=data[col].eq(level).to_numpy()
                estimates.append(group_statistics(data.loc[mask],w[mask]))
            for m in metrics:
                values=np.array([v.get(m,np.nan) for v in estimates])
                if np.isfinite(values).all(): draws[m].append(float(np.ptp(values)))
    gaps=[]
    for m in metrics:
        values=observed.loc[eligible,m].to_numpy() if eligible else np.array([])
        gap=float(np.ptp(values)) if len(values)>1 and np.isfinite(values).all() else np.nan
        b=draws[m]; lo,hi=np.quantile(b,[.025,.975]) if b else (np.nan,np.nan)
        gaps.append({"dimension":label,"quantity":m,"eligible_groups":len(eligible),
            "max_minus_min":gap,"CI_low":lo,"CI_high":hi,"valid_bootstrap":len(b),"requested_bootstrap":boot})
    return rows,gaps


def run(args):
    pred=select_predictions(pd.read_csv(args.predictions),args.model,args.route,args.calibrator,args.feature_set)
    frame=pd.read_csv(args.analysis_frame)
    data=pred.merge(frame[["row_id",*GROUPS.values()]],on="row_id",validate="one_to_one",how="left",indicator=True)
    if not data["_merge"].eq("both").all(): raise ValueError("部分预测学生找不到审计标签")
    if not 0<args.capacity<=1: raise ValueError("容量必须在(0,1]内")
    k=max(1,int(round(args.capacity*len(data))))
    data["selected"]=False
    order=data.sort_values(["probability","row_id"],ascending=[False,True],kind="stable").index[:k]
    data.loc[order,"selected"]=True
    rows=[]; gaps=[]
    for label,col in GROUPS.items():
        r,g=audit_dimension(data,col,label,args.min_n,args.min_events,args.bootstrap)
        rows.extend(r); gaps.extend(g)
    for school,part in data.groupby("institution"):
        rows.append({"dimension":"Institution","group":school,
            "status":"reportable" if len(part)>=args.min_n else "small_group",
            "institutions":1,**group_statistics(part,np.ones(len(part)))})
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name,records in [("Table_05_Group_and_Institution_Equity_Audit.csv",rows),
                         ("Table_05_Equity_Gap_Intervals.csv",gaps)]:
        table=pd.DataFrame(records)
        for field in ("model","route","feature_set","calibrator","capacity"): table[field]=getattr(args,field)
        table.to_csv(out/name,index=False,encoding="utf-8-sig")


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",default=str(ROOT/"artifacts/calibrated_outer_oof_predictions.csv"))
    p.add_argument("--analysis-frame",default=str(ROOT/"artifacts/analysis_frame.csv"))
    p.add_argument("--model",default="ESD_MTMS_Net_M1"); p.add_argument("--route",default="J",choices=["S","D","J"])
    p.add_argument("--feature-set",default="T0",choices=["T0","T2"]); p.add_argument("--calibrator",default="platt")
    p.add_argument("--capacity",type=float,default=.1); p.add_argument("--min-n",type=int,default=100)
    p.add_argument("--min-events",type=int,default=20); p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--output",default=str(HERE/"results"))
    return p.parse_args()
if __name__=="__main__": run(cli())
