"""固定支持容量下的P1全局排序与P2高校比例配额比较；只输出核心配置指标。"""
from __future__ import annotations
import argparse,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"03_probability_calibration"))
from probability_calibration import select_predictions

def largest_remainder(sizes: pd.Series,k: int)->dict:
    if k<0 or k>sizes.sum(): raise ValueError("K超出可分配学生人数")
    raw=sizes/sizes.sum()*k; base=np.floor(raw).astype(int); remain=k-int(base.sum())
    order=(raw-base).sort_values(ascending=False,kind="stable").index
    for school in order[:remain]: base.loc[school]+=1
    return base.to_dict()

def select_policy(df,k,policy):
    if df.index.duplicated().any() or k<1 or k>len(df): raise ValueError("名单索引或容量无效")
    original_index=df.index
    df=df.sort_values(["probability","row_id"],ascending=[False,True],kind="stable")
    chosen=pd.Series(False,index=df.index)
    if policy=="P1_global_top_k": chosen.loc[df.nlargest(k,"probability").index]=True
    elif policy=="P2_proportional_school_quota":
        quotas=largest_remainder(df.groupby("institution").size(),k)
        for school,q in quotas.items():
            if q: chosen.loc[df[df.institution==school].nlargest(q,"probability").index]=True
    else: raise ValueError(policy)
    if chosen.sum()!=k: raise AssertionError("策略未保持相同支持名额K")
    return chosen.reindex(original_index)

def metrics(df,selected):
    y=df.observed_risk.to_numpy(int); s=selected.to_numpy(bool); k=s.sum(); positives=y.sum()
    # Coverage only asks whether a university has at least one student in the
    # final list; no additional institution-level performance indicators are
    # calculated here.
    selected_by_institution=df.assign(selected=s).groupby("institution",sort=False)["selected"].sum()
    return {"K":int(k),"Precision_at_K":y[s].mean() if k else np.nan,
        "Recall_at_K":y[s].sum()/positives if positives else np.nan,
        "Institution_coverage":float((selected_by_institution>0).mean())}

def paired_cluster_bootstrap(d,capacity,repetitions,seed=42,workers=1):
    """以高校为聚类单位重抽样，比较P2-P1，避免把学生当独立复制。

    原实现每次重复均创建134个DataFrame并拼接，在2,000次重复时带来大量
    非统计性的内存和对象开销。这里保持相同的有放回高校抽样与P1/P2定义，
    仅将每所高校预先表示为概率与标签数组。每一次被抽中的高校副本视为一个
    独立的临时高校，从而与旧版的 ``bootstrap_j`` 重命名规则一致。
    """
    ordered=d.sort_values("row_id",kind="stable").reset_index(drop=True)
    blocks=[]
    for _,part in ordered.groupby("institution",sort=False):
        # P2 only ever needs the number of positives in the within-school
        # top-q list.  Cache that prefix sum once rather than sorting every
        # school again in every bootstrap draw.
        probability=part.probability.to_numpy(float)
        outcome=part.observed_risk.to_numpy(int)
        local_order=np.argsort(-probability,kind="stable")
        blocks.append((probability,outcome,np.r_[0,np.cumsum(outcome[local_order])]))
    if len(blocks)<2: raise ValueError("高校聚类重抽样至少需要两所高校")
    sizes=np.asarray([len(p) for p,_,_ in blocks],dtype=int)
    rng=np.random.default_rng(seed)
    # 在主线程一次性产生全部抽样索引，因而无论并行与否均保留同一个随机序列。
    all_draws=rng.integers(0,len(blocks),size=(repetitions,len(blocks)))
    def one_draw(draws):
        sample_sizes=sizes[draws]
        n=int(sample_sizes.sum()); k=max(1,int(round(capacity*n)))
        probability=np.concatenate([blocks[j][0] for j in draws])
        outcome=np.concatenate([blocks[j][1] for j in draws])
        offsets=np.r_[0,np.cumsum(sample_sizes)]
        # P1: 全局Top-K。argpartition 给出的恰是Top-K集合，但无需对其余
        # (n-k)名学生排序；风险分数为连续值，因而边界并列不会影响本研究结果。
        p1_index=np.argpartition(-probability,k-1)[:k]
        p1_positive=int(outcome[p1_index].sum())
        p1_covered=np.unique(np.searchsorted(offsets,p1_index,side="right")-1).size
        # P2: 按被抽中的每一个临时高校副本比例配额，再在副本内按风险排序。
        raw_quota=sample_sizes/n*k; quota=np.floor(raw_quota).astype(int)
        remainder=k-int(quota.sum())
        if remainder:
            quota[np.argsort(-(raw_quota-quota),kind="stable")[:remainder]]+=1
        if int(quota.sum())!=k:
            raise AssertionError("bootstrap样本未保持固定支持容量")
        positives=outcome.sum()
        # 已缓存的前缀和给出每个临时高校副本的校内Top-q命中数。
        p2_positive=int(sum(blocks[j][2][q] for j,q in zip(draws,quota)))
        difference_positive=p2_positive-p1_positive
        return np.asarray([
            difference_positive/positives if positives else np.nan,
            difference_positive/k,
            (np.count_nonzero(quota)-p1_covered)/len(draws),
        ])
    if workers>1:
        # 每次高校重抽样相互独立；线程只并行数值拼接与选择，不改变抽样、
        # 容量配额或置信区间的计算规则。
        with ThreadPoolExecutor(max_workers=workers) as executor:
            differences=np.asarray(list(executor.map(one_draw,all_draws)),dtype=float)
    else:
        differences=np.asarray([one_draw(draws) for draws in all_draws],dtype=float)
    metric_names=["Recall_at_K","Precision_at_K","Institution_coverage"]
    result=pd.DataFrame({"metric":metric_names,
        "CI_low":np.quantile(differences,.025,axis=0),
        "CI_high":np.quantile(differences,.975,axis=0)})
    k=max(1,int(round(capacity*len(d))))
    m1=metrics(d,select_policy(d,k,"P1_global_top_k")); m2=metrics(d,select_policy(d,k,"P2_proportional_school_quota"))
    result["point_estimate"]=result.metric.map({key:m2[key]-m1[key] for key in m1})
    return result

def run(args):
    d=select_predictions(pd.read_csv(args.predictions),args.model,args.route,args.calibrator,args.feature_set)
    if any(not 0<c<=1 for c in [*args.capacities,args.primary_capacity]) or args.bootstrap<1:
        raise ValueError("容量必须在(0,1]，bootstrap次数必须为正")
    if args.primary_capacity not in args.capacities: raise ValueError("主容量必须包含在容量列表中")
    sys.path.insert(0,str(ROOT/"04_social_fairness"))
    from social_fairness import GROUPS,group_statistics
    audit=pd.read_csv(args.analysis_frame)
    d=d.merge(audit[["row_id",*GROUPS.values()]],on="row_id",validate="one_to_one",how="left",indicator=True)
    if not d["_merge"].eq("both").all(): raise ValueError("分配审计标签缺失")
    rows=[]; decisions=[]; group_rows=[]
    for capacity in args.capacities:
        k=max(1,int(round(capacity*len(d))))
        for policy in ("P1_global_top_k","P2_proportional_school_quota"):
            chosen=select_policy(d,k,policy); rows.append({"capacity":capacity,"policy":policy,**metrics(d,chosen)})
            assigned=d.assign(selected=chosen)
            for dimension,column in GROUPS.items():
                for level,g in assigned.groupby(column,dropna=False):
                    summary=group_statistics(g,np.ones(len(g)))
                    group_rows.append({"capacity":capacity,"policy":policy,"dimension":dimension,
                        "group":level,"n":summary["n"],"events":summary["events"],"TPR":summary["TPR"]})
            if abs(capacity-args.primary_capacity)<1e-10:
                z=d[["row_id","institution","probability","observed_risk"]].copy(); z["policy"]=policy; z["selected"]=chosen.to_numpy(); decisions.append(z)
    out=Path(args.output); art=Path(args.artifact_dir); out.mkdir(parents=True,exist_ok=True); art.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"Table_06_Capacity_Allocation_Policies.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(group_rows).to_csv(out/"Table_06_Policy_Group_TPR.csv",index=False,encoding="utf-8-sig")
    ci=paired_cluster_bootstrap(d,args.primary_capacity,args.bootstrap,workers=args.bootstrap_workers)
    ci.insert(0,"capacity",args.primary_capacity); ci.to_csv(out/"Table_06_Paired_Policy_Intervals.csv",index=False,encoding="utf-8-sig")
    pd.concat(decisions,ignore_index=True).to_csv(art/"primary_capacity_allocation_decisions.csv",index=False,encoding="utf-8-sig")

def cli():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",default=str(ROOT/"artifacts/calibrated_outer_oof_predictions.csv")); p.add_argument("--feature-set",default="T0",choices=["T0","T2"]); p.add_argument("--analysis-frame",default=str(ROOT/"artifacts/analysis_frame.csv")); p.add_argument("--model",default="ESD_MTMS_Net_M1"); p.add_argument("--route",default="J"); p.add_argument("--calibrator",default="platt"); p.add_argument("--capacities",type=float,nargs="+",default=[.05,.10,.20,.30]); p.add_argument("--primary-capacity",type=float,default=.10); p.add_argument("--bootstrap",type=int,default=2000); p.add_argument("--bootstrap-workers",type=int,default=1,help="并行独立高校聚类重抽样的线程数；不改变随机抽样序列。"); p.add_argument("--output",default=str(HERE/"results")); p.add_argument("--artifact-dir",default=str(ROOT/"artifacts")); return p.parse_args()
if __name__=="__main__": run(cli())
