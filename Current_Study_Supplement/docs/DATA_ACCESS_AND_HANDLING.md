# Data access and handling

The raw student workbook and all student-level derived files are intentionally excluded from this public supplement. The source dataset and any required permissions or access conditions are described in the manuscript's Data Availability statement.

To reproduce the analyses, provide the authorized workbook locally at `code/data/source_workbook.xlsx` (or update `data_workbook` in `code/config/experiment_config.json`) and set the correct worksheet name. The preprocessing and modeling pipeline creates local analysis frames, fold assignments, OOF predictions, and optional explanation intermediates under `code/artifacts/`; these files are student-level and must remain private unless the data provider and applicable privacy rules permit their release.

The public results tables are aggregate summaries. Individual SHAP values and per-student OOF predictions are not included.
