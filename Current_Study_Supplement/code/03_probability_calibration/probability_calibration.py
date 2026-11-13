"""嵌套概率处理选择：训练高校内选择，再评价完全留出的高校。

S为负英语预测分数，D为分类原概率，J为联合模型logit。对每个外层高校
留出折，identity、Platt和Isotonic仅在外层训练高校的内部校准折之间比较
Brier分数；随后将被选中的处理器拟合到全部外层训练高校的交叉拟合预测，并
固定应用于外层测试高校。此文件不训练预测模型、不使用外层测试标签选择或
拟合概率处理器。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KEY = ["model", "feature_set", "route", "outer_fold"]
IDENTITY = [*KEY, "row_id"]
REQUIRED = [*IDENTITY, "institution", "seed", "observed_score", "predicted_score",
            "observed_risk", "risk_threshold", "raw_score"]


def validate_records(frame: pd.DataFrame, name: str) -> None:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}缺少字段: {sorted(missing)}")
    if frame.empty or frame.duplicated([*IDENTITY, "seed"]).any():
        raise ValueError(f"{name}为空或存在同学生/模型/路线/折/种子重复记录")
    if not set(frame.route).issubset({"S", "D", "J"}):
        raise ValueError("风险路线必须为S、D或J")
    if not np.isfinite(frame.raw_score.to_numpy(float)).all():
        raise ValueError("风险分数含非有限值")
    if not frame.observed_risk.isin([0, 1]).all():
        raise ValueError("风险标签必须为0/1")
    if not np.isfinite(frame[["observed_score","risk_threshold"]].to_numpy(float)).all():
        raise ValueError("结果分数或风险阈值非有限")
    expected_label=(frame.observed_score<=frame.risk_threshold).astype(int)
    if not np.array_equal(expected_label.to_numpy(),frame.observed_risk.to_numpy()):
        raise ValueError("风险标签不符合该折阈值定义")
    if not np.isfinite(frame.loc[frame.route.isin(["S","J"]),"predicted_score"]).all():
        raise ValueError("S/J路线缺少有效连续预测")
    for _, group in frame.groupby(KEY, dropna=False):
        expected = frozenset(group.seed.unique())
        actual = group.groupby("row_id").seed.agg(lambda x: frozenset(x))
        if not actual.map(lambda x: x == expected).all():
            raise ValueError(f"{name}学生之间种子覆盖不一致")
    stable = ["institution", "observed_score", "observed_risk", "risk_threshold"]
    if (frame.groupby(IDENTITY)[stable].nunique(dropna=False) > 1).any().any():
        raise ValueError(f"{name}同学生跨种子的真值或事件定义不一致")


def aggregate_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """固定算术平均：S在分数尺度，D在原概率尺度，J在logit尺度。"""
    result=frame.groupby(IDENTITY, as_index=False).agg(
        institution=("institution", "first"), observed_score=("observed_score", "first"),
        predicted_score=("predicted_score", "mean"), observed_risk=("observed_risk", "first"),
        risk_threshold=("risk_threshold", "first"), raw_score=("raw_score", "mean"),
        seed_count=("seed", "nunique"))
    if "institution_input_shift" in frame:
        drift=frame.groupby(IDENTITY,as_index=False).institution_input_shift.first()
        result=result.merge(drift,on=IDENTITY,validate="one_to_one")
    return result


def aggregate_inner_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """在保留内部校准折标识的前提下聚合随机种子。

    calibration_fold 是嵌套选择的验证单位，不能像旧版脚本那样在种子平均时
    被丢弃；否则会在同一份内部 OOF 记录上拟合并评价校准器。
    """
    if "calibration_fold" not in frame:
        raise ValueError("校准记录缺少calibration_fold，无法进行嵌套概率处理选择")
    group = [*IDENTITY, "calibration_fold"]
    result = frame.groupby(group, as_index=False).agg(
        institution=("institution", "first"), observed_score=("observed_score", "first"),
        predicted_score=("predicted_score", "mean"), observed_risk=("observed_risk", "first"),
        risk_threshold=("risk_threshold", "first"), raw_score=("raw_score", "mean"),
        seed_count=("seed", "nunique"))
    if "institution_input_shift" in frame:
        drift = frame.groupby(group, as_index=False).institution_input_shift.first()
        result = result.merge(drift, on=group, validate="one_to_one")
    if result.duplicated(group).any():
        raise ValueError("种子聚合后内部校准记录重复")
    return result


def select_predictions(frame,model,route,calibrator,feature_set):
    """下游严格选择一种实验口径；拒绝把不同特征集/重复学生混为独立结果。"""
    selected=frame[(frame.model==model)&(frame.route==route)&
        (frame.calibrator==calibrator)&(frame.feature_set==feature_set)].copy()
    if selected.empty: raise ValueError("没有指定模型、风险路线、校准器和特征集的预测")
    if selected.row_id.duplicated().any(): raise ValueError("下游要求每名学生唯一OOF预测，不能再次平均不同实验")
    if not np.isfinite(selected.calibrated_probability).all(): raise ValueError("预测概率非有限")
    return selected.rename(columns={"calibrated_probability":"probability"}).reset_index(drop=True)


def score_coordinate(x, route):
    x = np.asarray(x, dtype=float)
    if route == "D":
        if np.any((x < 0) | (x > 1)):
            raise ValueError("D路线原概率不在[0,1]")
        return logit(np.clip(x, 1e-7, 1 - 1e-7))
    return x


def native_probability(values, route):
    """将原始模型输出变换为合法概率；S路线只有排序分数，禁止伪造概率。"""
    values = np.asarray(values, dtype=float)
    if route == "D":
        if np.any((values < 0) | (values > 1)):
            raise ValueError("D路线原概率不在[0,1]")
        return np.clip(values, 1e-7, 1 - 1e-7)
    if route == "J":
        return np.clip(expit(values), 1e-7, 1 - 1e-7)
    raise ValueError("S路线没有原始概率，不能使用identity概率处理")


def fit_calibrator(kind, x, y, route):
    """返回单调预测函数与可复核参数。单一类别时拒绝伪造校准结果。"""
    if kind == "identity":
        if route == "S":
            raise ValueError("S路线没有可保留的原始概率")
        return (lambda values: native_probability(values, route),
                {"method": "identity", "fitted": False})
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("校准训练记录必须包含两类结果")
    x = score_coordinate(x, route)
    if kind == "isotonic":
        estimator = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        state = {"x_thresholds": estimator.X_thresholds_.tolist(),
                 "y_thresholds": estimator.y_thresholds_.tolist()}
        return lambda z: estimator.predict(score_coordinate(z, route)), state
    if kind != "platt":
        raise ValueError(kind)
    center, scale = float(np.mean(x)), max(float(np.std(x)), 1e-8)
    z = (x - center) / scale
    def objective(theta):
        eta = theta[0] + theta[1] * z
        error = expit(eta) - y
        return (float(np.mean(np.logaddexp(0, eta) - y * eta)),
                np.array([error.mean(), np.mean(error * z)]))
    fit = minimize(objective, [float(logit(y.mean())), 1.0], jac=True,
                   method="L-BFGS-B", bounds=[(None, None), (0.0, None)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Platt优化失败: {fit.message}")
    intercept, slope = map(float, fit.x)
    return (lambda values: expit(intercept + slope * (score_coordinate(values, route) - center) / scale),
            {"intercept": intercept, "nonnegative_slope": slope, "center": center, "scale": scale})


def candidate_methods(route: str) -> tuple[str, ...]:
    """S仅是排序分数；D/J可比较保留原概率与两种后校准。"""
    return ("platt", "isotonic") if route == "S" else ("identity", "platt", "isotonic")


def nested_select_method(training: pd.DataFrame, route: str) -> tuple[str, pd.DataFrame]:
    """在外层训练高校内对概率处理器进行折间交叉验证选择。

    对每个 calibration_fold，处理器在其余内部折的交叉拟合预测上拟合，
    仅在该折上计算 Brier。由全部内部验证折的拼接预测得到选择Brier，避免
    在同一批记录上拟合和评价可学习的 Platt/Isotonic 处理器。
    """
    folds = sorted(training.calibration_fold.dropna().unique().tolist())
    if len(folds) < 2:
        raise ValueError("内部校准至少需要两个高校分组折")
    records = []
    for method in candidate_methods(route):
        evaluated = []
        for fold in folds:
            fit = training[training.calibration_fold != fold]
            valid = training[training.calibration_fold == fold]
            if fit.empty or valid.empty:
                raise ValueError("内部校准折为空")
            predict, _ = fit_calibrator(method, fit.raw_score, fit.observed_risk, route)
            part = valid[["row_id", "observed_risk"]].copy()
            part["probability"] = predict(valid.raw_score)
            evaluated.append(part)
        checked = pd.concat(evaluated, ignore_index=True)
        if checked.row_id.duplicated().any() or len(checked) != len(training):
            raise ValueError("内部嵌套校准没有恰好覆盖每名训练学生一次")
        records.append({
            "candidate": method,
            "selection_n": int(len(checked)),
            "selection_events": int(checked.observed_risk.sum()),
            "selection_Brier": float(brier_score_loss(checked.observed_risk, checked.probability)),
        })
    table = pd.DataFrame(records)
    # 稳定的预先声明破同分规则：identity优先于Platt，Platt优先于Isotonic。
    priority = {name: index for index, name in enumerate(candidate_methods(route))}
    table["tie_priority"] = table.candidate.map(priority)
    chosen = table.sort_values(["selection_Brier", "tie_priority"], kind="stable").iloc[0].candidate
    table["selected"] = table.candidate.eq(chosen)
    return str(chosen), table.drop(columns="tie_priority")


def probability_metrics(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    both = len(np.unique(y)) == 2
    prevalence=float(y.mean()) if len(y) else np.nan
    ap=average_precision_score(y, p) if both else np.nan
    # AP equals PR-AUC under the non-interpolated average-precision definition.
    # APSS is explicitly defined here relative to the no-skill prevalence
    # baseline; it is retained for the pre-specified threshold sensitivity
    # analysis, not as a substitute for reporting AP itself.
    apss=(ap-prevalence)/(1-prevalence) if both and prevalence < 1 else np.nan
    return {"n": len(y), "events": int(y.sum()), "event_prevalence":prevalence,
            "PR_AUC": ap, "AP_Skill_Score":apss,
            "ROC_AUC": roc_auc_score(y, p) if both else np.nan,
            "Brier": brier_score_loss(y, p)}


def reliability_bins(frame, count=10):
    p = frame.calibrated_probability
    if p.nunique() < 2:
        labels = pd.Series("constant_prediction", index=frame.index)
    else:
        labels = pd.qcut(p, q=min(count, p.nunique()), duplicates="drop").astype(str)
    return frame.assign(bin=labels).groupby("bin", as_index=False, observed=True).agg(
        predicted_probability=("calibrated_probability", "mean"),
        observed_proportion=("observed_risk", "mean"), n=("row_id", "size"),
        institutions=("institution", "nunique"))


def run(args):
    raw = pd.read_csv(args.raw); inner = pd.read_csv(args.inner)
    validate_records(raw, "outer"); validate_records(inner, "calibration")
    outer_avg, inner_avg = aggregate_seeds(raw), aggregate_inner_seeds(inner)
    calibrated, states, selection_tables = [], [], []
    for keys, outer in outer_avg.groupby(KEY):
        mask = np.ones(len(inner_avg), dtype=bool)
        for column, value in zip(KEY, keys):
            mask &= inner_avg[column].eq(value).to_numpy()
        training = inner_avg.loc[mask]
        if training.empty:
            raise ValueError(f"缺少校准训练记录: {keys}")
        raw_seeds=set(raw.loc[(raw[KEY]==pd.Series(keys,index=KEY)).all(axis=1),"seed"])
        inner_seeds=set(inner.loc[(inner[KEY]==pd.Series(keys,index=KEY)).all(axis=1),"seed"])
        if raw_seeds!=inner_seeds: raise ValueError("校准集与测试集种子集合不一致")
        if set(training.institution) & set(outer.institution):
            raise ValueError("外层测试高校出现在校准训练记录中")
        if set(training.risk_threshold) != set(outer.risk_threshold):
            raise ValueError("校准与外层评价使用了不同事件阈值")
        seed_raw = raw
        for column, value in zip(KEY, keys):
            seed_raw = seed_raw[seed_raw[column] == value]
        selected_kind, selection = nested_select_method(training, keys[2])
        for column, value in zip(KEY, keys):
            selection[column] = value
        selection_tables.append(selection[[*KEY, "candidate", "selection_n", "selection_events", "selection_Brier", "selected"]])
        # 先保留所有候选处理的外层预测，便于正文的敏感性表与补充材料复核。
        candidate_results = {}
        for kind in candidate_methods(keys[2]):
            predict, state = fit_calibrator(kind, training.raw_score, training.observed_risk, keys[2])
            result = outer.copy(); result["calibrator"] = kind; result["seed"] = -1
            result["calibrated_probability"] = predict(outer.raw_score)
            result["selected_method"] = selected_kind
            seed_p = seed_raw.assign(p=predict(seed_raw.raw_score))
            disagreement = seed_p.groupby("row_id").p.std(ddof=0)
            result["seed_disagreement"] = result.row_id.map(disagreement)
            calibrated.append(result)
            candidate_results[kind] = result
            states.append(dict(zip(KEY, keys), calibrator=kind, selected=(kind == selected_kind), **state))
        # 唯一下游口径：在当前外层折中由内部嵌套选择确定的方法。
        selected_result = candidate_results[selected_kind].copy()
        selected_result["calibrator"] = "nested_selected"
        selected_result["selected_method"] = selected_kind
        calibrated.append(selected_result)
    cal = pd.concat(calibrated, ignore_index=True)
    rows, bins = [], []
    grouping = ["model", "feature_set", "route", "calibrator"]
    # 校准前后都在同一份外层OOF学生上评价。S只有排序分数，禁止伪造原概率。
    for keys,group in outer_avg.groupby(["model","feature_set","route"]):
        metadata=dict(zip(["model","feature_set","route"],keys))
        for fold,part in [("all",group),*list(group.groupby("outer_fold"))]:
            if keys[2]=="S":
                both=part.observed_risk.nunique()==2
                metrics={"n":len(part),"events":int(part.observed_risk.sum()),
                    "PR_AUC":average_precision_score(part.observed_risk,part.raw_score) if both else np.nan,
                    "ROC_AUC":roc_auc_score(part.observed_risk,part.raw_score) if both else np.nan}
            else:
                prob=part.raw_score if keys[2]=="D" else expit(part.raw_score)
                metrics=probability_metrics(part.observed_risk,prob)
            rows.append({**metadata,"calibrator":"uncalibrated","scope":"overall" if fold=="all" else "outer_fold",
                         "outer_fold":fold,**metrics})
    for keys, group in cal.groupby(grouping):
        metadata = dict(zip(grouping, keys))
        rows.append({**metadata, "scope": "overall", "outer_fold": "all",
                     **probability_metrics(group.observed_risk, group.calibrated_probability)})
        for fold, part in group.groupby("outer_fold"):
            rows.append({**metadata, "scope": "outer_fold", "outer_fold": fold,
                         **probability_metrics(part.observed_risk, part.calibrated_probability)})
        b = reliability_bins(group)
        for k, v in metadata.items(): b[k] = v
        bins.append(b)
    out, art = Path(args.output), Path(args.artifact_dir)
    out.mkdir(parents=True, exist_ok=True); art.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "Table_04_Risk_Discrimination_and_Calibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(selection_tables, ignore_index=True).to_csv(
        out / "Table_04_Nested_Calibration_Selection.csv", index=False, encoding="utf-8-sig")
    pd.concat(bins, ignore_index=True).to_csv(out / "Table_04_Calibration_Curve_Data.csv", index=False, encoding="utf-8-sig")
    cal.to_csv(art / "calibrated_outer_oof_predictions.csv", index=False, encoding="utf-8-sig")
    (out / "Manifest_03_Calibration.json").write_text(json.dumps({
        "aggregation": "arithmetic mean of raw score before calibration; D probability, J logit, S negative score",
        "calibration_target": "outer training threshold; no outer labels used in fitting",
        "candidate_methods": {"D/J": ["identity", "platt", "isotonic"], "S": ["platt", "isotonic"]},
        "selection_rule": "Within each outer fold, select the lowest Brier score from calibration-fold cross-validation; ties follow identity, Platt, Isotonic.",
        "downstream_probability": "nested_selected", "fitted_states": states},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=str(ROOT / "artifacts/raw_outer_oof_predictions.csv"))
    p.add_argument("--inner", default=str(ROOT / "artifacts/calibration_inner_oof_predictions.csv"))
    p.add_argument("--output", default=str(HERE / "results"))
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts"))
    return p.parse_args()

if __name__ == "__main__": run(cli())
