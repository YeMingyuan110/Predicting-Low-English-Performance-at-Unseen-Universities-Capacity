"""统一的高校隔离嵌套验证、调参与OOF预测入口。

路线S：连续成绩模型的负预测值作为低表现排序分数；路线D：直接二分类器；
路线J：ESD-MTMS-Net M1联合输出。风险阈值仅由当前外层训练集计算。
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict, ParameterSampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "01_data_preprocessing"))
from baseline_models import available_model_ids, build_spec, positive_class_probability
from data_preprocessing import (CONFIG_PATH, INSTITUTION, TARGET, SAMPLE_ID, TorchFoldPreprocessor,
                                feature_columns, load_config, prepare_project_data)
from esd_mtms_net import ESDConfig, fit_esd, predict_esd, train_esd_fixed


def regression_metrics(y, p):
    return {"RMSE": mean_squared_error(y, p) ** .5, "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p)}

def inner_cv(groups: pd.Series, n_splits: int):
    n = min(n_splits, groups.nunique())
    if n < 2: raise ValueError("内层验证至少需要两个高校")
    return GroupKFold(n_splits=n)


def tune_baseline(model_id, task, x, y, groups, feature_set, seed, n_iter, risk_quantile=.25):
    from data_preprocessing import make_sklearn_preprocessor
    spec = build_spec(model_id, task, make_sklearn_preprocessor(feature_set), seed)
    if task == "classification":
        # 每个调参验证折采用其训练高校定义的事件，不使用验证高校确定阈值。
        candidates=list(ParameterSampler(spec.search_space,n_iter=min(n_iter,int(np.prod([len(v) for v in spec.search_space.values()]))),random_state=seed))
        evaluated=[]
        for params in candidates:
            losses=[]
            for fit,valid in inner_cv(groups,3).split(x,groups=groups):
                threshold=float(x.iloc[fit][TARGET].quantile(risk_quantile))
                yf=(x.iloc[fit][TARGET].to_numpy()<=threshold).astype(int)
                yv=(x.iloc[valid][TARGET].to_numpy()<=threshold).astype(int)
                if len(np.unique(yf))!=2: raise ValueError("内层训练高校缺少风险类别")
                estimator=clone(spec.estimator).set_params(**params).fit(x.iloc[fit],yf)
                losses.append(brier_score_loss(yv,positive_class_probability(estimator,x.iloc[valid])))
            evaluated.append((float(np.mean(losses)),params))
        loss,params=min(evaluated,key=lambda item:item[0])
        fitted=clone(spec.estimator).set_params(**params).fit(x,y)
        return fitted,params,-loss
    scoring = "neg_root_mean_squared_error" if task == "regression" else "average_precision"
    search = RandomizedSearchCV(spec.estimator, spec.search_space, n_iter=min(n_iter,
        int(np.prod([len(v) for v in spec.search_space.values()]))), scoring=scoring,
        cv=inner_cv(groups, 3), random_state=seed, n_jobs=1, refit=True, error_score="raise")
    search.fit(x, y, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def baseline_outer_predictions(frame, train_idx, test_idx, model_id, feature_set, seed, n_iter, risk_quantile=.25):
    tr, te = frame.iloc[train_idx], frame.iloc[test_idx]
    xtr, xte, ytr = tr, te, tr[TARGET].to_numpy(float)
    threshold = float(np.quantile(ytr, risk_quantile))
    rtr, rte = (ytr <= threshold).astype(int), (te[TARGET].to_numpy(float) <= threshold).astype(int)
    # ESD/TabM may leave cached CUDA blocks after a completed fold.  CatBoost owns
    # a separate GPU allocator, so release inactive PyTorch cache before each of
    # its independent regression/classification searches.  This changes neither
    # data, split, seed nor model hyperparameters.
    def release_torch_cuda_cache():
        if model_id != "catboost":
            return
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            # CatBoost remains available on a CPU-only installation; cleanup is
            # merely a resource-management step and must not change the protocol.
            pass
    release_torch_cuda_cache()
    reg, reg_params, reg_cv = tune_baseline(model_id, "regression", xtr, ytr, tr[INSTITUTION], feature_set, seed, n_iter)
    pred = reg.predict(xte)
    release_torch_cuda_cache()
    cls, cls_params, cls_cv = tune_baseline(model_id, "classification", xtr, rtr, tr[INSTITUTION], feature_set, seed, n_iter,risk_quantile)
    prob = positive_class_probability(cls, xte)
    outer = pd.DataFrame({"row_id": te["row_id"], "institution": te[INSTITUTION], "outer_fold": te["outer_fold"],
        "seed": seed, "model": model_id, "feature_set": feature_set, "observed_score": te[TARGET],
        "risk_threshold": threshold, "observed_risk": rte, "predicted_score": pred,
        "route_S_raw_score": -pred, "route_D_raw_score": prob})
    # 校准样本由独立嵌套交叉拟合函数产生；此处不复用选参层预测。
    inner = None
    meta = {"regression_best_params": reg_params, "classification_best_params": cls_params,
            "regression_inner_score": reg_cv, "classification_inner_score": cls_cv}
    return outer, inner, meta


def _theory_groups(prep, feature_set):
    """按预先定义的教育构念映射数值/类别token，不按结果重新分组。"""
    numeric_dim=len(prep.numeric_scaler.mean_)
    if feature_set=="T0": return [list(range(numeric_dim))]
    blocks=[]; start=numeric_dim
    for card in prep.cardinalities:
        blocks.append([start]); start+=1
    flatten=lambda ids:[v for j in ids for v in blocks[j]]
    return [list(range(min(5,numeric_dim))),list(range(5,numeric_dim)),flatten(range(0,3)),
            flatten(range(3,12)),flatten(range(12,19))]

def _torch_matrix(prep, frame):
    numeric, categorical = prep.transform(frame)
    # 类别列仅作为embedding的查表索引；网络不会按连续值处理这些整数。
    # 未见类别使用折内预留的最后一类；类别词表从不在外层测试折拟合。
    return np.concatenate([numeric, categorical.astype(np.float32)], axis=1)

def esd_candidate_configs(max_epochs):
    """冻结的小型候选集；候选数量不随外层测试结果扩张。"""
    base = [
        ESDConfig(d_model=64,n_heads=4,n_layers=2,dropout=.15,topk_ratio=.50,max_epochs=max_epochs),
        ESDConfig(d_model=96,n_heads=4,n_layers=2,dropout=.10,topk_ratio=.50,max_epochs=max_epochs,learning_rate=5e-4),
        ESDConfig(d_model=64,n_heads=4,n_layers=3,dropout=.20,topk_ratio=.35,max_epochs=max_epochs,risk_weight=.50),
    ]
    # 十二项预先固定候选，与树模型搜索处于同一数量级。
    # 候选在看外层结果前固定；不能单独扩大ESD的预算以追求排名。
    from dataclasses import replace
    return [replace(c,learning_rate=lr,rank_weight=weight)
            for c in base for lr in (5e-4,1e-3) for weight in (.0,.05)]


def esd_outer_predictions(frame, train_idx, test_idx, feature_set, seed, mode, cfgs, risk_quantile=.25):
    tr, te = frame.iloc[train_idx], frame.iloc[test_idx]
    threshold = float(np.quantile(tr[TARGET], risk_quantile))
    y = tr[TARGET].to_numpy(float); risk = (y <= threshold).astype(float)
    splits=list(inner_cv(tr[INSTITUTION],3).split(tr,groups=tr[INSTITUTION]))
    candidates=[]
    for candidate_no,cfg in enumerate(cfgs):
        oof_score=np.full(len(tr),np.nan); oof_risk=np.full(len(tr),np.nan); fold_summaries=[]
        for inner_no,(fit_i,val_i) in enumerate(splits):
            prep=TorchFoldPreprocessor(feature_set).fit(tr.iloc[fit_i])
            cfg.categorical_cardinalities=tuple(prep.cardinalities)
            xfit,xval=_torch_matrix(prep,tr.iloc[fit_i]),_torch_matrix(prep,tr.iloc[val_i])
            y_mean,y_sd=y[fit_i].mean(),max(y[fit_i].std(),1e-8)
            # 内层训练标签只使用内层训练高校结果定义；不窥视内层验证标签。
            inner_threshold=float(np.quantile(y[fit_i],risk_quantile))
            inner_risk_train=(y[fit_i]<=inner_threshold).astype(float)
            inner_risk_valid=(y[val_i]<=inner_threshold).astype(float)
            selected,summary=fit_esd(xfit,(y[fit_i]-y_mean)/y_sd,inner_risk_train,xval,
                (y[val_i]-y_mean)/y_sd,inner_risk_valid,_theory_groups(prep,feature_set),cfg,mode,seed+inner_no)
            pvz,pvr=predict_esd(selected,xval); pvy=pvz*y_sd+y_mean
            if pvr is None: pvr=1/(1+np.exp((pvy-threshold)/max(y_sd,1e-8)))
            if mode=="M0": summary["best_validation_score"]*=y_sd
            oof_score[val_i]=pvy; oof_risk[val_i]=pvr; fold_summaries.append(summary)
        if np.isnan(oof_score).any(): raise RuntimeError("ESD内层OOF预测未覆盖全部外层训练学生")
        mean_rmse=float(np.mean([s["best_validation_score"] for s in fold_summaries]))
        candidates.append({"score":mean_rmse,"cfg":cfg,"oof_score":oof_score,"oof_risk":oof_risk,
            "epochs":[s["best_epoch"] for s in fold_summaries],"candidate_no":candidate_no})
    chosen=min(candidates,key=lambda z:z["score"]); cfg=chosen["cfg"]
    selected_epochs=max(1,int(round(np.median(chosen["epochs"]))))
    from dataclasses import asdict
    summary={"best_epoch":selected_epochs,"best_validation_score":chosen["score"],
        "selection_metric":"Brier" if mode=="M1" else "RMSE_original_score",
        "candidate_validation_scores":[c["score"] for c in candidates],
        "selected_candidate":chosen["candidate_no"],"selected_config":asdict(cfg),
        "inner_best_epochs":chosen["epochs"]}
    # 固定三折内层选择出的配置与轮次，在完整外层训练集上从头重训。
    prep_full=TorchFoldPreprocessor(feature_set).fit(tr); xfull=_torch_matrix(prep_full,tr); xtest=_torch_matrix(prep_full,te)
    cfg.categorical_cardinalities=tuple(prep_full.cardinalities)
    full_mean,full_sd=y.mean(),max(y.std(),1e-8)
    summary["selected_config"]=asdict(cfg)
    summary["outcome_scaling"]={"mean":float(full_mean),"sd":float(full_sd)}
    model=train_esd_fixed(xfull,(y-full_mean)/full_sd,risk,_theory_groups(prep_full,feature_set),cfg,mode,seed,selected_epochs)
    pz, pr = predict_esd(model, xtest); pred = pz * full_sd + full_mean
    if mode == "M0": pr = 1 / (1 + np.exp((pred-threshold) / max(full_sd, 1e-8)))
    rte = (te[TARGET].to_numpy(float) <= threshold).astype(int)
    name = f"ESD_MTMS_Net_{mode}"
    outer = pd.DataFrame({"row_id": te["row_id"], "institution": te[INSTITUTION], "outer_fold": te["outer_fold"],
        "seed": seed, "model": name, "feature_set": feature_set, "observed_score": te[TARGET],
        "risk_threshold": threshold, "observed_risk": rte, "predicted_score": pred,
        "route_S_raw_score": -pred, "route_D_raw_score": pr})
    # 校准使用覆盖完整外层训练集的高校隔离内层OOF预测。
    inner = pd.concat([pd.DataFrame({"outer_fold": te["outer_fold"].iloc[0], "seed": seed, "model": name,
        "feature_set": feature_set, "route": "S", "observed_risk": risk.astype(int), "raw_score": -chosen["oof_score"]}),
        pd.DataFrame({"outer_fold": te["outer_fold"].iloc[0], "seed": seed, "model": name,
        "feature_set": feature_set, "route": "J" if mode=="M1" else "D", "observed_risk": risk.astype(int), "raw_score": chosen["oof_risk"]})])
    return outer, inner, summary


def independent_calibration_predictions(frame, train_idx, fold, model_id, feature_set, seed,
                                       n_iter, risk_quantile, cfgs=None, mode=None):
    """独立高校交叉拟合：校准留出高校不参与超参数、预处理或早停选择。

    校准目标始终是外层训练人口定义的固定事件Y<=q_outer。子模型在各自
    拟合高校定义风险学习标签；校准以其留出分数映射到固定的外层目标。
    这是一项明确的分数到目标事件的映射，不能将子模型原概率称为最终概率。
    """
    train=frame.iloc[train_idx].reset_index(drop=True)
    target_threshold=float(train[TARGET].quantile(risk_quantile)); records=[]
    for calibration_fold,(fit,holdout) in enumerate(inner_cv(train[INSTITUTION],3).split(train,groups=train[INSTITUTION]),1):
        if set(train.iloc[fit][INSTITUTION]) & set(train.iloc[holdout][INSTITUTION]):
            raise AssertionError("校准高校泄漏")
        # 训练函数内部重新选择超参数和轮次；弃用其选参层OOF输出。
        if mode is None:
            predictions,_,_=baseline_outer_predictions(train,fit,holdout,model_id,feature_set,seed,n_iter,risk_quantile)
        else:
            import copy
            predictions,_,_=esd_outer_predictions(train,fit,holdout,feature_set,seed,mode,copy.deepcopy(cfgs),risk_quantile)
        for route,column in [("S","route_S_raw_score"),("J" if mode=="M1" else "D","route_D_raw_score")]:
            if mode=="M0" and route=="D": continue
            records.append(pd.DataFrame({"row_id":predictions.row_id.to_numpy(),
                "institution":predictions.institution.to_numpy(),"outer_fold":fold,
                "calibration_fold":calibration_fold,"seed":seed,"model":model_id,
                "feature_set":feature_set,"route":route,"risk_threshold":target_threshold,
                "observed_risk":(predictions.observed_score.to_numpy()<=target_threshold).astype(int),
                "observed_score":predictions.observed_score.to_numpy(),
                "predicted_score":predictions.predicted_score.to_numpy() if route!="D" else np.nan,
                "raw_score":(np.log(np.clip(predictions[column].to_numpy(),1e-7,1-1e-7)/(1-np.clip(predictions[column].to_numpy(),1e-7,1-1e-7))) if route=="J" else predictions[column].to_numpy())}))
    result=pd.concat(records,ignore_index=True)
    if result.duplicated(["row_id","route"]).any(): raise AssertionError("校准样本重复")
    return result


def summarize(outer):
    """主表先平均同一学生的多种子连续预测，再计算一次指标。"""
    rows=[]
    for (model,fs), g in outer.groupby(["model","feature_set"]):
        ensemble=g.groupby(["row_id","outer_fold"],as_index=False).agg(
            observed_score=("observed_score","first"),predicted_score=("predicted_score","mean"))
        rows.append({"model":model,"feature_set":fs,"scope":"ensemble_overall",
                     "n":len(ensemble),**regression_metrics(ensemble.observed_score,ensemble.predicted_score)})
        for fold,part in ensemble.groupby("outer_fold"):
            rows.append({"model":model,"feature_set":fs,"scope":"ensemble_outer_fold",
                "outer_fold":fold,"n":len(part),**regression_metrics(part.observed_score,part.predicted_score)})
    return pd.DataFrame(rows)


def paired_regression_intervals(outer,repetitions=2000):
    """固定OOF预测的高校配对bootstrap；不是对训练过程重新抽样。

    点估计来自原样本。高校抽样次数作为权重，所有比较模型共享同一抽样。
    报告ESD减基线的差异，不能把包含零的区间写成已证实优越。
    """
    if repetitions<1: raise ValueError("bootstrap次数必须为正")
    rows=[]
    for fs,part in outer.groupby("feature_set"):
        avg=part.groupby(["model","row_id","institution"],as_index=False).agg(
            y=("observed_score","first"),pred=("predicted_score","mean"))
        for main in ("ESD_MTMS_Net_M0","ESD_MTMS_Net_M1"):
            a=avg[avg.model==main]
            if a.empty: continue
            for baseline in sorted(set(avg.model)-{main}):
                b=avg[avg.model==baseline]
                paired=a.merge(b,on=["row_id","institution"],suffixes=("_a","_b"),validate="one_to_one")
                if len(paired)!=len(a) or len(paired)!=len(b): raise ValueError("模型比较学生覆盖不同")
                if not np.allclose(paired.y_a,paired.y_b): raise ValueError("模型比较标签不一致")
                paired=paired.assign(n=1,y=paired.y_a,y2=paired.y_a**2,
                    sse_a=(paired.pred_a-paired.y_a)**2,sse_b=(paired.pred_b-paired.y_a)**2,
                    sae_a=abs(paired.pred_a-paired.y_a),sae_b=abs(paired.pred_b-paired.y_a))
                stats=paired.groupby("institution")[["n","y","y2","sse_a","sse_b","sae_a","sae_b"]].sum().to_numpy()
                weights=np.random.default_rng(42).multinomial(len(stats),np.full(len(stats),1/len(stats)),size=repetitions)
                total=np.vstack([stats.sum(axis=0),weights@stats]); n,y,y2,sa,sb,aa,ab=total.T
                variance=y2-y*y/n
                delta={"RMSE":np.sqrt(sa/n)-np.sqrt(sb/n),"MAE":(aa-ab)/n,
                       "R2":np.divide(sb-sa,variance,out=np.full_like(sa,np.nan),where=variance>0)}
                for metric,values in delta.items():
                    valid=values[1:][np.isfinite(values[1:])]
                    lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
                    rows.append({"feature_set":fs,"model":main,"comparator":baseline,"metric":metric,
                        "difference_model_minus_comparator":values[0],"CI_low":lo,"CI_high":hi,
                        "valid_bootstrap":len(valid),"institutions":len(stats)})
    return pd.DataFrame(rows)


def _checkpoint_dir(art, args, cfg):
    """隔离不同数据、代码和实验协议的暂存结果，避免错误地复用旧预测。"""
    # A supplied directory is an explicit, auditable request to resume a
    # verified interrupted run.  This is useful when only downstream
    # reporting code changes; it never changes a model, split or training
    # configuration.
    if args.checkpoint_dir:
        path=Path(args.checkpoint_dir)
        path.mkdir(parents=True,exist_ok=True)
        return path
    identity={"analysis_sha256":hashlib.sha256(Path(args.analysis_frame).read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_module_sha256":{
                  name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                  for name in ("baseline_models.py","esd_mtms_net.py","esd_original_adapted.py")},
              "config":cfg,"feature_set":args.feature_set,"models":args.models,
              "n_iter":args.n_iter,"esd_epochs":args.esd_epochs,
              "esd_trials":args.esd_trials,"risk_quantile":args.risk_quantile,
              "training_backend":args.training_backend,"max_rows":args.max_rows}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()[:16]
    path=art/f"_incomplete_{digest}"
    path.mkdir(parents=True,exist_ok=True)
    return path


def _task_result(checkpoint_dir, fold, model_id, seed, producer):
    """逐任务持久化，意外中断后可从最后一个完整高校折继续。"""
    name=f"fold_{int(fold):02d}_{model_id}_seed_{int(seed)}"
    outer_path=checkpoint_dir/f"{name}_outer.csv"
    inner_path=checkpoint_dir/f"{name}_inner.csv"
    meta_path=checkpoint_dir/f"{name}_meta.json"
    if all(p.exists() for p in (outer_path,inner_path,meta_path)):
        outer=pd.read_csv(outer_path,encoding="utf-8-sig")
        inner=pd.read_csv(inner_path,encoding="utf-8-sig")
        metadata=json.loads(meta_path.read_text(encoding="utf-8"))
        if not (outer.outer_fold.eq(fold).all() and outer.model.eq(model_id).all()
                and outer.seed.eq(seed).all() and inner.outer_fold.eq(fold).all()
                and inner.model.eq(model_id).all() and inner.seed.eq(seed).all()):
            raise ValueError(f"暂存结果身份不一致: {name}")
        print(f"[resume] {name}",flush=True)
        return outer,inner,metadata
    print(f"[start] {name}",flush=True)
    outer,inner,metadata=producer()
    for path,writer in ((outer_path,lambda p:outer.to_csv(p,index=False,encoding="utf-8-sig")),
                        (inner_path,lambda p:inner.to_csv(p,index=False,encoding="utf-8-sig")),
                        (meta_path,lambda p:p.write_text(json.dumps(metadata,ensure_ascii=False,default=str),encoding="utf-8"))):
        temporary=path.with_suffix(path.suffix+".tmp")
        writer(temporary)
        os.replace(temporary,path)
    print(f"[done] {name}: test_n={len(outer)} calibration_n={len(inner)}",flush=True)
    return outer,inner,metadata


def run(args):
    # 正式训练读取已经审计的数据和固定高校折，不重新清洗或随机拆分学生。
    cfg=load_config(Path(args.config)); out=Path(args.output); art=Path(args.artifact_dir)
    frame=pd.read_csv(args.analysis_frame,encoding="utf-8-sig",keep_default_na=False,na_values=[""])
    if frame.row_id.duplicated().any() or not frame.groupby(INSTITUTION).outer_fold.nunique().eq(1).all():
        raise ValueError("学生标识或高校分折无效")
    if not np.isfinite(frame[TARGET]).all(): raise ValueError("目标分数含非有限值")
    if not 0<args.risk_quantile<1: raise ValueError("分位数必须在0至1之间")
    if args.n_iter<1 or args.esd_epochs<1: raise ValueError("搜索次数和训练轮次必须为正")
    if int(cfg["inner_folds"])!=3: raise ValueError("当前冻结实现要求三折内层验证")
    if not args.max_rows and frame.outer_fold.nunique()!=int(cfg["outer_folds"]):
        raise ValueError("外层折数与协议不一致")
    if args.max_rows: frame=frame.sample(min(args.max_rows,len(frame)),random_state=42).reset_index(drop=True); from data_preprocessing import assign_outer_folds; frame["outer_fold"]=assign_outer_folds(frame,min(5,frame[INSTITUTION].nunique()))
    models=args.models.split(",") if args.models else available_model_ids(False)
    from data_preprocessing import make_sklearn_preprocessor
    for model_id in models:
        for task in ("regression","classification"):
            build_spec(model_id,task,make_sklearn_preprocessor(args.feature_set),42)
    if args.check_only:
        print("PREFLIGHT_OK",len(frame),frame[INSTITUTION].nunique(),models); return
    out.mkdir(parents=True,exist_ok=True); art.mkdir(parents=True,exist_ok=True)
    if (art/"raw_outer_oof_predictions.csv").exists() and not args.overwrite: raise FileExistsError("已有结果；请指定新输出目录或--overwrite")
    checkpoint_dir=_checkpoint_dir(art,args,cfg)
    outer_all=[]; inner_all=[]; metadata=[]
    for fold in sorted(frame.outer_fold.unique()):
        print(f"[outer fold {int(fold)}/{frame.outer_fold.nunique()}]",flush=True)
        tri=np.flatnonzero(frame.outer_fold.to_numpy()!=fold); tei=np.flatnonzero(frame.outer_fold.to_numpy()==fold)
        if set(frame.iloc[tri][INSTITUTION])&set(frame.iloc[tei][INSTITUTION]):
            raise AssertionError("外层训练/测试高校重叠")
        # 无标签漂移：以训练高校插补/标准化后的输入为参照，计算各测试高校
        # 数值特征均值偏移的RMS。它是描述信号，不是正式分布检验或p值。
        drift_prep=TorchFoldPreprocessor(args.feature_set).fit(frame.iloc[tri])
        train_num,_=drift_prep.transform(frame.iloc[tri]); test_num,_=drift_prep.transform(frame.iloc[tei])
        center=train_num.mean(axis=0); drift={}
        schools=frame.iloc[tei][INSTITUTION].to_numpy()
        for school in np.unique(schools):
            drift[school]=float(np.sqrt(np.mean((test_num[schools==school].mean(axis=0)-center)**2)))
        # 传统基线只跑预注册固定种子42；TabM与ESD保留三种子集成，
        # 因为它们的神经网络初始化和批次顺序会改变结果。
        for model_id in models:
            model_seeds = cfg["random_seeds"] if model_id=="tabm" else [cfg["random_seeds"][0]]
            for seed in model_seeds:
                def produce_baseline():
                    o,i,m=baseline_outer_predictions(frame,tri,tei,model_id,args.feature_set,seed,args.n_iter,args.risk_quantile)
                    i=independent_calibration_predictions(frame,tri,int(fold),model_id,args.feature_set,seed,args.n_iter,args.risk_quantile)
                    o["institution_input_shift"]=o.institution.map(drift)
                    return o,i,{"outer_fold":int(fold),"seed":seed,"model":model_id,**m}
                o,i,m=_task_result(checkpoint_dir,fold,model_id,seed,produce_baseline)
                outer_all.append(o); inner_all.append(i); metadata.append(m)
        # 主模型独立于基线循环；每个外层折、模式和预先固定种子只训练一次。
        for mode in ("M0","M1"):
            candidates=esd_candidate_configs(args.esd_epochs)[:args.esd_trials]
            for c in candidates:
                c.patience=min(15,max(2,args.esd_epochs//4)); c.backend=args.training_backend
            model_name=f"ESD_MTMS_Net_{mode}"
            for seed in cfg["random_seeds"]:
                def produce_esd():
                    import copy
                    local_candidates=copy.deepcopy(candidates)
                    o,i,m=esd_outer_predictions(frame,tri,tei,args.feature_set,seed,mode,local_candidates,args.risk_quantile)
                    i=independent_calibration_predictions(frame,tri,int(fold),model_name,args.feature_set,seed,args.n_iter,args.risk_quantile,local_candidates,mode)
                    o["institution_input_shift"]=o.institution.map(drift)
                    return o,i,{"outer_fold":int(fold),"seed":seed,"model":model_name,
                        **{k:v for k,v in m.items() if k!="history"}}
                o,i,m=_task_result(checkpoint_dir,fold,model_name,seed,produce_esd)
                outer_all.append(o); inner_all.append(i); metadata.append(m)
    outer=pd.concat(outer_all,ignore_index=True); inner=pd.concat(inner_all,ignore_index=True)
    # 下游统一使用长表S/D/J；M0不存在直接分类路线。
    long_records=[]
    for model_id,g in outer.groupby("model"):
        for route,column in [("S","route_S_raw_score"),("J" if model_id.endswith("M1") else "D","route_D_raw_score")]:
            if model_id.endswith("M0") and route=="D": continue
            z=g.drop(columns=["route_S_raw_score","route_D_raw_score"]).copy(); z["route"]=route
            p=np.clip(g[column].to_numpy(),1e-7,1-1e-7)
            z["raw_score"]=np.log(p/(1-p)) if route=="J" else g[column].to_numpy()
            if route=="D": z["predicted_score"]=np.nan
            long_records.append(z)
    output_long=pd.concat(long_records,ignore_index=True)
    if output_long.duplicated(["model","feature_set","route","seed","row_id"]).any():
        raise AssertionError("学生OOF重复")
    if not output_long.groupby(["model","feature_set","route","seed"]).row_id.nunique().eq(len(frame)).all():
        raise AssertionError("OOF未覆盖全部学生")
    fold_audit=[]
    for fold,g in frame.groupby("outer_fold"):
        training=frame[frame.outer_fold!=fold]
        q=float(training[TARGET].quantile(args.risk_quantile))
        fold_audit.append({"outer_fold":fold,"test_students":len(g),"test_institutions":g[INSTITUTION].nunique(),
            "train_students":len(training),"train_institutions":training[INSTITUTION].nunique(),
            "risk_threshold":q,"test_events":int((g[TARGET]<=q).sum())})
    output_long.to_csv(art/"raw_outer_oof_predictions.csv",index=False,encoding="utf-8-sig"); inner.to_csv(art/"calibration_inner_oof_predictions.csv",index=False,encoding="utf-8-sig")
    summarize(outer).to_csv(out/"Table_03_Model_Prediction_Performance.csv",index=False,encoding="utf-8-sig")
    paired_regression_intervals(outer,int(cfg["bootstrap_repetitions"])).to_csv(
        out/"Table_03_Paired_Model_Comparison.csv",index=False,encoding="utf-8-sig")
    import hashlib, platform, importlib.metadata
    versions={}
    for package in ("numpy","pandas","scipy","scikit-learn","torch","lightgbm","xgboost","catboost","tabm"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    manifest={"arguments":vars(args),"protocol":cfg,"python":platform.python_version(),
        "gpu_training_implementation":"CUDA-resident minibatches and device-side shuffling when CUDA is available",
        "package_versions":versions,"analysis_sha256":hashlib.sha256(Path(args.analysis_frame).read_bytes()).hexdigest(),
        "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.rglob("*.py")},
        "selection_records":metadata,"fold_audit":fold_audit,
        "calibration":"nested group cross-fitting; fixed outer-training event",
        "smoke_only":bool(args.max_rows)}
    (out/"model_validation_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    shutil.rmtree(checkpoint_dir)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(CONFIG_PATH)); p.add_argument("--feature-set",default="T0",choices=["T0","T2"])
    p.add_argument("--checkpoint-dir",default=None,
                   help="已核验的中断训练检查点目录；仅用于恢复同一数据、划分和训练配置。")
    p.add_argument("--models",default="linear,random_forest,lightgbm,xgboost,catboost"); p.add_argument("--n-iter",type=int,default=12)
    p.add_argument("--analysis-frame",default=str(ROOT/"artifacts/analysis_frame.csv"))
    p.add_argument("--check-only",action="store_true"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--training-backend",default="torch",choices=["torch","lightning"])
    p.add_argument("--risk-quantile",type=float,default=.25,
                   help="主分析为0.25；0.15与0.35仅作预设阈值敏感性分析。")
    p.add_argument("--esd-epochs",type=int,default=120); p.add_argument("--esd-trials",type=int,default=12,choices=range(1,13)); p.add_argument("--max-rows",type=int)
    p.add_argument("--output",default=str(HERE/"results")); p.add_argument("--artifact-dir",default=str(ROOT/"artifacts")); return p.parse_args()
if __name__=="__main__": run(parse_args())
