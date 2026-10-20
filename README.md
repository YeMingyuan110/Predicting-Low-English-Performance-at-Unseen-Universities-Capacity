# Supplementary Code Package

This package contains the selected analysis code, model settings, software-environment records, and run provenance supporting the manuscript. Student-level data, trained weights, generated results, figures, notebooks, and manuscript files are not included.

## Contents

```text
11_Submission_Code_Package/
├── code/
│   ├── 01_data_preprocessing/
│   ├── 02_measurement_and_suitability/
│   ├── 03_main_model/
│   ├── 04_baseline_comparison/
│   ├── 05_ablation_and_task_interaction/
│   ├── 06_cross_university_generalization/
│   ├── 07_capacity_constrained_screening/
│   ├── 08_fairness_and_sdg4/
│   └── 09_interpretability_and_application/
├── config/
├── environment/
├── provenance/
└── docs/
```

## Key files

- Study protocol: `config/study_protocol.yaml`
- Main-model parameters: `config/main_model_m2.yaml`
- Baseline parameters: `config/baseline_models.yaml`
- Software environments: `environment/ENVIRONMENT_NOTES.md`
- File-purpose mapping: `docs/FILE_MANIFEST.md`
- Required external inputs: `docs/EXPECTED_INPUTS.md`
- Execution sequence: `docs/EXECUTION_GUIDE.md`
- Integrity checksums: `provenance/SHA256SUMS.csv`

## Data and model files

No student-level data are redistributed. The source dataset should be obtained from the repository cited in the manuscript. Running the complete workflow additionally requires the fixed university-fold assignment. The interpretation analysis also requires the fitted fold/seed checkpoints and their preprocessing states.

## Environment records

The main neural model and the strong baselines were executed in different recorded software environments. Use the separate requirement files under `environment/`; the machine-generated JSON files under `provenance/` remain the authoritative historical records.

## Naming note

Directory names, expected input names, and generated output names in this submission copy were standardized in English. These edits affect paths and filenames only; the analytical procedures were not changed. Consequently, source-code hashes embedded in the historical run manifests refer to the original execution copies, whereas `provenance/SHA256SUMS.csv` records the files distributed in this package.

## Public release

Before depositing this package in a public repository, add the final author metadata, repository citation, persistent identifier, and an appropriate software licence. Do not upload student-level inputs or predictions unless their redistribution is permitted.

