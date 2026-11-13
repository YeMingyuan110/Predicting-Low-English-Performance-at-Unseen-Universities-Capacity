"""Cross-validated SHAP explanation for the M1-J English low-performance risk model.

Reconstructs the already selected outer-fold M1 models from the saved validation
manifest, verifies their predictions against archived OOF predictions, and then
explains the uncalibrated low-performance probability on held-out universities.
No held-out labels are used to fit preprocessing, models, or explanations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_model_prediction"))
sys.path.insert(0, str(ROOT / "01_data_preprocessing"))

from data_preprocessing import INSTITUTION, TARGET, TorchFoldPreprocessor, feature_columns
from esd_mtms_net import ESDConfig, predict_esd, train_esd_fixed
from validation_and_tuning import _theory_groups, _torch_matrix


FEATURE_LABELS = {
    "高中数学成绩": ("Prior mathematics achievement", "高中数学成绩"),
    "高中批判性阅读成绩": ("Prior critical-reading achievement", "高中批判性阅读成绩"),
    "高中公民素养成绩": ("Prior civic-competence achievement", "高中公民素养成绩"),
    "高中生物成绩": ("Prior biology achievement", "高中生物成绩"),
    "高中英语成绩": ("Prior English achievement", "高中英语成绩"),
}


class RiskProbability(torch.nn.Module):
    """Expose one scalar output: M1's probability of the fold-defined risk event."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x)["risk_logit"]).unsqueeze(-1)


def exact_shapley_single_reference(model, x, baseline, chunk_size=64):
    """Enumerate all 2^M coalitions for exact SHAP values with one reference row."""
    n_rows, n_features = x.shape
    n_coalitions = 1 << n_features
    masks = torch.tensor(
        [[(coalition >> j) & 1 for j in range(n_features)] for coalition in range(n_coalitions)],
        dtype=x.dtype,
        device=x.device,
    )
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, n_rows, chunk_size):
            part = x[start:start + chunk_size]
            mixed = baseline.unsqueeze(1) + (part.unsqueeze(1) - baseline.unsqueeze(1)) * masks.unsqueeze(0)
            flat = mixed.reshape(-1, n_features)
            values = []
            for batch_start in range(0, len(flat), 4096):
                values.append(model(flat[batch_start:batch_start + 4096]).squeeze(-1))
            outputs.append(torch.cat(values).reshape(len(part), n_coalitions).cpu().numpy())
    coalition_values = np.concatenate(outputs, axis=0).astype(np.float64, copy=False)
    import math
    factorial = [math.factorial(i) for i in range(n_features + 1)]
    denominator = factorial[n_features]
    shap_values = np.zeros((n_rows, n_features), dtype=np.float64)
    for feature in range(n_features):
        bit = 1 << feature
        for coalition in range(n_coalitions):
            if coalition & bit:
                continue
            coalition_size = int(coalition.bit_count())
            weight = factorial[coalition_size] * factorial[n_features - coalition_size - 1] / denominator
            shap_values[:, feature] += weight * (
                coalition_values[:, coalition | bit] - coalition_values[:, coalition]
            )
    base_value = float(coalition_values[0, 0])
    predictions = coalition_values[:, -1]
    residual = predictions - (base_value + shap_values.sum(axis=1))
    return shap_values, base_value, predictions, residual

def _draw_explanation_sample(test_frame, max_per_university, seed):
    """Deterministic university-stratified sample; retain every test university."""
    pieces = []
    for _, group in test_frame.groupby(INSTITUTION, sort=True):
        n = min(max_per_university, len(group))
        pieces.append(group.sample(n=n, random_state=seed) if n < len(group) else group)
    selected = pd.concat(pieces).sort_values("row_id")
    counts = selected.groupby(INSTITUTION).size()
    population = test_frame.groupby(INSTITUTION).size()
    selected = selected.copy()
    selected["sampling_weight"] = selected[INSTITUTION].map(population / counts)
    return selected


def run(args):
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; refusing a silent CPU run.")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    analysis_path = Path(args.analysis_frame)
    manifest_path = Path(args.validation_manifest)
    oof_path = Path(args.raw_oof)
    output_dir = Path(args.output)
    for path in (analysis_path, manifest_path, oof_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(analysis_path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oof = pd.read_csv(oof_path, encoding="utf-8-sig")
    records = [r for r in manifest["selection_records"] if r.get("model") == "ESD_MTMS_Net_M1"]
    if len(records) != 15:
        raise ValueError(f"Expected 5 folds × 3 M1 seeds in manifest, found {len(records)}")
    if frame["row_id"].duplicated().any():
        raise ValueError("analysis_frame contains duplicate row_id")
    if frame.groupby(INSTITUTION)["outer_fold"].nunique().ne(1).any():
        raise ValueError("A university is split across outer folds")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA disappeared during setup.")

    numeric, categorical = feature_columns("T0")
    if categorical:
        raise ValueError("This explainability implementation expects the prespecified T0 numeric inputs.")
    feature_names = [FEATURE_LABELS.get(name, (name, name))[0] for name in numeric]
    feature_names_zh = [name for name in numeric]
    shap_rows, run_records = [], []
    reference_failures = []

    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"SHAP: {shap.__version__}; rows={len(frame)}; universities={frame[INSTITUTION].nunique()}", flush=True)
    print(f"Selected M1 models to reconstruct: {len(records)}", flush=True)

    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train_frame = frame.loc[frame["outer_fold"].astype(int) != fold].copy()
        test_frame = frame.loc[frame["outer_fold"].astype(int) == fold].copy()
        explain_frame = _draw_explanation_sample(
            test_frame, args.max_per_university, args.sample_seed + fold
        )
        print(
            f"[fold {fold}/5] train_students={len(train_frame)} "
            f"test_universities={test_frame[INSTITUTION].nunique()} "
            f"explained_rows={len(explain_frame)}",
            flush=True,
        )

        prep = TorchFoldPreprocessor("T0").fit(train_frame)
        x_train = _torch_matrix(prep, train_frame)
        x_test = _torch_matrix(prep, test_frame)
        x_explain = _torch_matrix(prep, explain_frame)
        y_train = train_frame[TARGET].to_numpy(float)
        threshold = float(np.quantile(y_train, args.risk_quantile))
        risk_train = (y_train <= threshold).astype(float)
        y_mean = float(y_train.mean())
        y_sd = max(float(y_train.std()), 1e-8)
        groups = _theory_groups(prep, "T0")

        # A single training-only mean profile is the reference point. T0 has five
        # continuous inputs, so exact enumeration needs only 2^5=32 coalitions.
        baseline = torch.as_tensor(x_train.mean(axis=0, keepdims=True), dtype=torch.float32, device=device)
        explain_tensor = torch.as_tensor(x_explain, dtype=torch.float32, device=device)

        fold_records = [r for r in records if int(r["outer_fold"]) == fold]
        if len(fold_records) != 3:
            raise ValueError(f"Fold {fold} does not have exactly three saved seed records.")
        for rec in sorted(fold_records, key=lambda z: int(z["seed"])):
            seed = int(rec["seed"])
            cfg_dict = dict(rec["selected_config"])
            cfg_dict["categorical_cardinalities"] = tuple(cfg_dict.get("categorical_cardinalities", ()))
            cfg = ESDConfig(**cfg_dict)
            epochs = int(rec["best_epoch"])
            model = train_esd_fixed(
                x_train,
                (y_train - y_mean) / y_sd,
                risk_train,
                groups,
                cfg,
                "M1",
                seed,
                epochs,
            )
            model.eval()
            p_score, probability = predict_esd(model, x_test)
            if probability is None:
                raise RuntimeError(f"M1 seed {seed}, fold {fold} did not return a risk probability.")

            archived = oof[
                (oof["model"] == "ESD_MTMS_Net_M1")
                & (oof["feature_set"] == "T0")
                & (oof["route"] == "J")
                & (oof["outer_fold"].astype(int) == fold)
                & (oof["seed"].astype(int) == seed)
            ][["row_id", "raw_score"]].copy()
            # The archived long-form OOF table stores route-J raw_score as log-odds.
            # Convert it back to probability before comparing with predict_esd().
            archived["archived_probability"] = 1.0 / (1.0 + np.exp(-archived["raw_score"].clip(-40, 40)))
            archived = archived[["row_id", "archived_probability"]]
            current = pd.DataFrame({"row_id": test_frame["row_id"].to_numpy(), "reconstructed": probability})
            comparison = current.merge(archived, on="row_id", validate="one_to_one")
            if len(comparison) != len(test_frame):
                raise ValueError(f"Stored OOF coverage mismatch for fold {fold}, seed {seed}")
            diff = np.abs(comparison["reconstructed"].to_numpy() - comparison["archived_probability"].to_numpy())
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            parity_ok = max_diff <= args.prediction_tolerance
            if not parity_ok:
                reference_failures.append(
                    {"fold": fold, "seed": seed, "max_abs_prediction_difference": max_diff}
                )
                raise RuntimeError(
                    f"Reconstructed model does not match archived OOF prediction "
                    f"(fold={fold}, seed={seed}, max_abs_diff={max_diff:.6g}). "
                    "Stopping before SHAP to avoid explaining a different model."
                )
            print(f"  seed={seed}: fixed_epoch={epochs}; OOF max_abs_diff={max_diff:.3g}", flush=True)

            wrapped = RiskProbability(model).eval()
            values, base_value, predicted, residual = exact_shapley_single_reference(
                wrapped, explain_tensor, baseline, chunk_size=args.explain_batch_size
            )
            # Preserve the SHAP explanation object and check its additive reconstruction.
            explanation = shap.Explanation(
                values=values,
                base_values=np.full(len(values), base_value),
                data=explain_frame[numeric].to_numpy(dtype=float),
                feature_names=feature_names,
                output_names=["M1-J low-performance risk probability"],
            )
            reconstructed = explanation.base_values + explanation.values.sum(axis=1)
            if not np.allclose(reconstructed, predicted, rtol=1e-6, atol=1e-7):
                raise RuntimeError("Exact SHAP coalition values failed additivity validation.")
            residual = predicted - reconstructed

            for i, (_, row) in enumerate(explain_frame.iterrows()):
                record = {
                    "row_id": int(row["row_id"]),
                    "outer_fold": fold,
                    "seed": seed,
                    "institution_population_n": int((test_frame[INSTITUTION] == row[INSTITUTION]).sum()),
                    "sampled_from_institution_n": int((explain_frame[INSTITUTION] == row[INSTITUTION]).sum()),
                    "sampling_weight": float(row["sampling_weight"]),
                    "observed_english_score": float(row[TARGET]),
                    "observed_low_performance": int(float(row[TARGET]) <= threshold),
                    "outer_training_risk_threshold": threshold,
                    "risk_probability": float(predicted[i]),
                    "base_value": base_value,
                    "additivity_residual": float(residual[i]),
                    "oof_prediction_max_abs_difference": max_diff,
                }
                for j, name in enumerate(numeric):
                    record[f"input__{name}"] = float(row[name]) if pd.notna(row[name]) else np.nan
                    record[f"shap__{name}"] = float(values[i, j])
                shap_rows.append(record)
            run_records.append({
                "outer_fold": fold,
                "seed": seed,
                "best_epoch_from_original_validation": epochs,
                "selected_candidate": rec.get("selected_candidate"),
                "max_abs_oof_prediction_difference": max_diff,
                "mean_abs_oof_prediction_difference": mean_diff,
                "n_explained": len(explain_frame),
                "reference_profile": "outer-training mean on standardized T0 inputs",
                "exact_coalitions": 1 << len(feature_names),
                "additivity_mean_abs_residual": float(np.mean(np.abs(residual))),
                "additivity_p95_abs_residual": float(np.quantile(np.abs(residual), 0.95)),
                "additivity_max_abs_residual": float(np.max(np.abs(residual))),
            })
            del model, wrapped, explanation
            torch.cuda.empty_cache()

    raw = pd.DataFrame(shap_rows)
    id_cols = [
        "row_id", "outer_fold", "institution_population_n", "sampled_from_institution_n",
        "sampling_weight", "observed_english_score", "observed_low_performance",
        "outer_training_risk_threshold",
    ]
    input_cols = [f"input__{name}" for name in numeric]
    attribution_cols = [f"shap__{name}" for name in numeric]
    # Average attributions and predictions over the same three fixed seeds for each student.
    aggregate = raw.groupby(id_cols, as_index=False, dropna=False).agg(
        risk_probability=("risk_probability", "mean"),
        base_value=("base_value", "mean"),
        additivity_residual=("additivity_residual", "mean"),
        **{column: (column, "first") for column in input_cols},
        **{column: (column, "mean") for column in attribution_cols},
    )
    weights = aggregate["sampling_weight"].to_numpy(float)
    importance = []
    for name, label, zh, col in zip(numeric, feature_names, feature_names_zh, attribution_cols):
        vals = aggregate[col].to_numpy(float)
        importance.append({
            "feature": label,
            "feature_zh": zh,
            "mean_abs_shap_probability": float(np.average(np.abs(vals), weights=weights)),
            "mean_abs_shap_percentage_points": float(100 * np.average(np.abs(vals), weights=weights)),
            "mean_signed_shap_probability": float(np.average(vals, weights=weights)),
            "mean_signed_shap_percentage_points": float(100 * np.average(vals, weights=weights)),
            "interpretation": "positive values raise the model-estimated probability of low English performance; negative values lower it",
            "n_students_explained": int(len(aggregate)),
            "n_universities_represented": int(frame[INSTITUTION].nunique()),
        })
    importance = pd.DataFrame(importance).sort_values(
        "mean_abs_shap_probability", ascending=False
    ).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_dir / "Student_Level_SHAP_Attributions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output_dir / "Table_SHAP_Global_Importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(run_records).to_csv(output_dir / "SHAP_Run_Audit.csv", index=False, encoding="utf-8-sig")

    # Publication-ready two-panel global summary. SHAP values are in risk-probability units.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.6), gridspec_kw={"width_ratios": [0.9, 1.5]})
    plot_order = importance.sort_values("mean_abs_shap_probability", ascending=True)
    ax1.barh(
        plot_order["feature"],
        plot_order["mean_abs_shap_percentage_points"],
        color="#d95f59",
        edgecolor="#8f3533",
        linewidth=0.5,
    )
    ax1.set_xlabel("Mean |SHAP value| (percentage points)")
    ax1.set_title("(a) Global importance")
    ax1.grid(axis="x", linestyle=":", alpha=0.35)
    ax1.set_axisbelow(True)

    # Beeswarm-like display: deterministic jitter, colored by the observed input value.
    sorted_names = importance["feature"].tolist()
    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(args.sample_seed)
    for rank, label in enumerate(sorted_names):
        name = numeric[feature_names.index(label)]
        shap_col = f"shap__{name}"
        input_col = f"input__{name}"
        xv = aggregate[shap_col].to_numpy(float) * 100
        raw_feature = aggregate[input_col].to_numpy(float)
        lo, hi = np.nanquantile(raw_feature, [0.02, 0.98])
        color_values = np.clip((raw_feature - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.uniform(-0.22, 0.22, size=len(xv))
        ax2.scatter(xv, rank + jitter, c=color_values, cmap=cmap, vmin=0, vmax=1,
                    s=11, alpha=0.62, linewidths=0, rasterized=True)
    ax2.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax2.set_yticks(np.arange(len(sorted_names)))
    ax2.set_yticklabels(sorted_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("SHAP value (percentage points in predicted risk)")
    ax2.set_title("(b) Direction and distribution")
    ax2.grid(axis="x", linestyle=":", alpha=0.35)
    ax2.set_axisbelow(True)
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax2, fraction=0.035, pad=0.03)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Observed feature value")
    fig.suptitle("Cross-university SHAP analysis of low English-performance risk", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_SHAP_Global_Interpretation.png", bbox_inches="tight")
    fig.savefig(output_dir / "Figure_SHAP_Global_Interpretation.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest_out = {
        "analysis_question": "Which T0 prior-achievement features contribute to M1-J's predicted low-English-performance risk?",
        "explained_model": "ESD-MTMS-Net M1-J; original selected configuration and fixed epoch reconstructed per outer fold and seed",
        "output_explained": "uncalibrated M1-J risk probability for the fold-specific q25 low-performance event",
        "calibrated_probabilities_used_downstream": True,
        "explanation_method": "Exact Shapley enumeration over all feature subsets (2^5 coalitions)",
        "reference_profile": "mean of the corresponding outer-training standardized T0 inputs",
        "sample_design": f"up to {args.max_per_university} held-out students per university, all held-out universities retained",
        "aggregation": "per-student SHAP values averaged over seeds 42, 3407, 2027; global importance weighted back to held-out university student counts",
        "random_seeds": sorted({int(r["seed"]) for r in records}),
        "outer_folds": 5,
        "risk_quantile": args.risk_quantile,
        "feature_set": "T0",
        "features": [{"english": FEATURE_LABELS.get(name, (name, name))[0], "original": name} for name in numeric],
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "shap": shap.__version__,
        "student_rows_explained_after_seed_aggregation": len(aggregate),
        "universities_represented": int(frame[INSTITUTION].nunique()),
        "prediction_parity_tolerance": args.prediction_tolerance,
        "prediction_parity_failures": reference_failures,
        "weighted_mean_absolute_additivity_residual": float(np.average(
            np.abs(aggregate["additivity_residual"].to_numpy(float)), weights=weights
        )),
        "elapsed_seconds": time.time() - started,
        "interpretation_limit": "Attributions describe model behavior under the training-data background and are not causal effects. Correlated prior achievement measures can share attribution.",
        "privacy_note": "Student_Level_SHAP_Attributions.csv contains pseudonymous row IDs and student-level explanations; keep it in restricted project storage and do not publish it as open supplementary data.",
    }
    (output_dir / "SHAP_Run_Manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed: {len(aggregate)} unique explained students from {manifest_out['universities_represented']} universities.", flush=True)
    print(f"Global SHAP table: {output_dir / 'Table_SHAP_Global_Importance.csv'}", flush=True)
    print(f"Figure: {output_dir / 'Figure_SHAP_Global_Interpretation.png'}", flush=True)
    print(f"Elapsed seconds: {manifest_out['elapsed_seconds']:.1f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-frame", default=str(ROOT / "artifacts" / "analysis_frame.csv"))
    p.add_argument("--validation-manifest", default=str(ROOT / "02_model_prediction" / "results" / "T0_q0.25" / "model_validation_manifest.json"))
    p.add_argument("--raw-oof", default=str(ROOT / "artifacts" / "T0_q0.25" / "raw_outer_oof_predictions.csv"))
    p.add_argument("--output", default=str(ROOT / "02_model_prediction" / "results" / "T0_q0.25" / "Explainability_SHAP"))
    p.add_argument("--risk-quantile", type=float, default=0.25)
    p.add_argument("--max-per-university", type=int, default=20)
    p.add_argument("--explain-batch-size", type=int, default=64)
    p.add_argument("--sample-seed", type=int, default=20261112)
    p.add_argument("--prediction-tolerance", type=float, default=0.002)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

