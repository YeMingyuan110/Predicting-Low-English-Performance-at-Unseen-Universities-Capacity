from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'code' / 'artifacts' / 'T0_q0.25' / 'nested_calibration_selection'
PRED_PATH = BASE / 'calibrated_outer_oof_predictions.csv'
CURVE_PATH = PROJECT / 'code' / '03_probability_calibration' / 'results' / 'T0_q0.25' / 'nested_calibration_selection' / 'Table_04_Calibration_Curve_Data.csv'
OUT = PROJECT / 'results' / 'additional_risk_comparison'
TARGET = ('ESD_MTMS_Net_M1', 'J')
COMPARATORS = [('tabm', 'D'), ('lightgbm', 'D'), ('linear', 'D')]
LABELS = {'ESD_MTMS_Net_M1': 'M1-J', 'tabm': 'TabM', 'lightgbm': 'LightGBM', 'linear': 'Linear model'}
BOOTSTRAPS = 5000
SEED = 20261113


def metrics(y, p):
    ap = average_precision_score(y, p)
    prevalence = float(np.mean(y))
    ap_skill = (ap - prevalence) / (1.0 - prevalence) if prevalence < 1 else np.nan
    return ap, brier_score_loss(y, p), ap_skill


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_PATH)
    selection = pred[
        ((pred.model == TARGET[0]) & (pred.route == TARGET[1])) |
        (pred.model.isin([m for m, _ in COMPARATORS]) & (pred.route == 'D'))
    ].query("calibrator == 'nested_selected'").copy()
    selection['institution'] = selection['institution'].astype(str)
    if selection.duplicated(['model', 'route', 'row_id']).any():
        raise ValueError('Duplicate student predictions found within a model-route.')
    needed = [TARGET, *COMPARATORS]
    groups = {(m, r): g.set_index('row_id', drop=False) for m, r in needed
              for g in [selection[(selection.model == m) & (selection.route == r)]]}
    for key, g in groups.items():
        if g.empty:
            raise ValueError(f'Missing matched model predictions: {key}')
    reference = groups[TARGET].sort_index()
    for key, g in groups.items():
        g = g.sort_index()
        if not np.array_equal(reference.index.to_numpy(), g.index.to_numpy()):
            raise ValueError(f'OOF row IDs not aligned for {key}')
        for col in ['institution', 'observed_risk', 'outer_fold']:
            if not np.array_equal(reference[col].astype(str).to_numpy(), g[col].astype(str).to_numpy()):
                raise ValueError(f'Mismatched {col} for {key}')
    if reference.institution.nunique() != 134:
        raise ValueError(f"Expected 134 held-out universities; observed {reference.institution.nunique()}")
    y = reference.observed_risk.to_numpy(dtype=int)
    probability_col = 'calibrated_probability'
    point = {key: metrics(y, g.sort_index()[probability_col].to_numpy(float)) for key, g in groups.items()}

    # Paired cluster bootstrap: resample universities with replacement and carry
    # the same sampled student rows into every model, preserving model pairing.
    school_indices = {school: np.flatnonzero(reference.institution.to_numpy() == school)
                      for school in reference.institution.drop_duplicates()}
    schools = list(school_indices)
    rng = np.random.default_rng(SEED)
    boot = { (m, r): [] for m, r in COMPARATORS }
    target_p = groups[TARGET].sort_index()[probability_col].to_numpy(float)
    baseline_ps = {key: groups[key].sort_index()[probability_col].to_numpy(float) for key in COMPARATORS}
    for _ in range(BOOTSTRAPS):
        draw = rng.integers(0, len(schools), size=len(schools))
        idx = np.concatenate([school_indices[schools[j]] for j in draw])
        yy = y[idx]
        ap_t, br_t, skill_t = metrics(yy, target_p[idx])
        for key, pp in baseline_ps.items():
            ap_b, br_b, skill_b = metrics(yy, pp[idx])
            # Positive deltas consistently favor M1-J: AP/AP skill M1-J minus baseline;
            # Brier baseline minus M1-J because lower Brier is better.
            boot[key].append((ap_t - ap_b, br_b - br_t, skill_t - skill_b))

    metric_specs = [
        ('PR-AUC', 0, 'M1-J minus comparator'),
        ('Brier score', 1, 'Comparator minus M1-J'),
        ('AP Skill Score', 2, 'M1-J minus comparator'),
    ]
    rows = []
    target_vals = point[TARGET]
    for key in COMPARATORS:
        base_vals = point[key]
        deltas = np.asarray(boot[key])
        point_deltas = (target_vals[0] - base_vals[0], base_vals[1] - target_vals[1], target_vals[2] - base_vals[2])
        for name, col, direction in metric_specs:
            lo, hi = np.quantile(deltas[:, col], [0.025, 0.975])
            rows.append({
                'comparison': f"M1-J vs {LABELS[key[0]]}", 'metric': name,
                'M1J_estimate': target_vals[col if col != 1 else 1],
                'comparator_estimate': base_vals[col if col != 1 else 1],
                'paired_difference': point_deltas[col], 'difference_definition': direction,
                'CI_95_low': lo, 'CI_95_high': hi, 'bootstrap_replicates': BOOTSTRAPS,
                'cluster_unit': 'university', 'universities': len(schools), 'students': len(y),
                'interpretation': 'CI includes 0: no clear difference' if lo <= 0 <= hi else ('M1-J favored' if lo > 0 else 'Comparator favored')
            })
    pd.DataFrame(rows).to_csv(OUT / 'Table_Risk_Model_Paired_Cluster_Bootstrap.csv', index=False, encoding='utf-8-sig')

    # ECE computed from the existing nested-selected, equal-frequency reliability bins.
    curve = pd.read_csv(CURVE_PATH)
    curve = curve[
        ((curve.model == TARGET[0]) & (curve.route == TARGET[1])) |
        (curve.model.isin([m for m, _ in COMPARATORS]) & (curve.route == 'D'))
    ].query("calibrator == 'nested_selected'").copy()
    ece_rows = []
    for (model, route), g in curve.groupby(['model', 'route'], sort=False):
        n = g.n.to_numpy(float)
        ece = float(np.average(np.abs(g.observed_proportion - g.predicted_probability), weights=n))
        ece_rows.append({'model': LABELS[model], 'route': route, 'calibrator': 'nested_selected',
                         'ECE': ece, 'students': int(n.sum()), 'bins': len(g),
                         'binning': 'existing equal-frequency bins from Table_04_Calibration_Curve_Data.csv'})
    ece_df = pd.DataFrame(ece_rows).sort_values('model', key=lambda s: s.map({'M1-J':0,'TabM':1,'LightGBM':2,'Linear model':3}))
    ece_df.to_csv(OUT / 'Table_Calibration_ECE.csv', index=False, encoding='utf-8-sig')

    # Publication-ready reliability diagram. Curves use the same existing
    # 10-bin OOF summaries; identity line indicates perfect calibration.
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    ax.plot([0, 1], [0, 1], '--', color='#555555', lw=1.2, label='Perfect calibration')
    styles = {
        'M1-J': ('#1768AC', 'o', '-'), 'TabM': ('#D97706', 's', '-'),
        'LightGBM': ('#198754', '^', '-'), 'Linear model': ('#8A4F7D', 'D', '-')
    }
    for _, row in ece_df.iterrows():
        lab = row['model']
        g = curve[curve.model.map(LABELS) == lab].sort_values('predicted_probability')
        color, marker, ls = styles[lab]
        ax.plot(g.predicted_probability, g.observed_proportion, marker=marker, ls=ls,
                color=color, lw=1.7, ms=4.5, label=f"{lab} (ECE={row.ECE:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted risk', ylabel='Observed low-performance proportion')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#D9D9D9', lw=0.6, alpha=0.8)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#777777', fontsize=8.5)
    ax.spines[['top','right','bottom','left']].set_visible(True)
    fig.tight_layout()
    fig.savefig(OUT / 'Figure_Risk_Calibration_Reliability.png', dpi=600, bbox_inches='tight')
    fig.savefig(OUT / 'Figure_Risk_Calibration_Reliability.pdf', bbox_inches='tight')
    plt.close(fig)

    # Short audit file with explicit caveats and copy-ready summary values.
    pd.DataFrame([{'model': LABELS[k[0]], 'PR_AUC': v[0], 'Brier': v[1], 'AP_Skill_Score': v[2]}
                  for k, v in point.items()]).to_csv(OUT / 'Table_Risk_Model_Point_Estimates.csv', index=False, encoding='utf-8-sig')
    print(f'Wrote outputs to {OUT}')
    print(f'Matched OOF: {len(y)} students; {len(schools)} universities; events={int(y.sum())}; prevalence={y.mean():.6f}')
    print(pd.DataFrame(rows)[['comparison','metric','paired_difference','CI_95_low','CI_95_high','interpretation']].to_string(index=False))
    print(ece_df[['model','ECE','students','bins']].to_string(index=False))

if __name__ == '__main__':
    main()



