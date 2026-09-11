> **Historical validation notice:** the counts below describe the frozen submitted 46-run protocol. The active reviewer-ready baseline validation is documented in `FINAL_REVIEWER_REVISION_CODE_REPORT.md`, `BASELINE_FIDELITY.md`, and the 177-test suite.

# Paper baseline and reference-control validation

- Strict method IDs: 10
- Strict literature-baseline runs: 12
- Corrected reference-control runs: 20 (`full`, `linear`)
- Full TRSO proposal runs: 10
- Separate TRSO ablations: 4
- Total training runs: 46 across six Kaggle sessions
- Full repository tests: 150 passed
- Full Fine-Tuning one-epoch fair-suite smoke: passed; all model parameters trainable
- Linear Probing one-epoch fair-suite smoke: passed; task head only trainable
- ARC structural/gradient preflight: passed
- ARC exact merge: 4 branches folded, maximum absolute output error 0.0 on the preflight model
- Removed transferred/engineering controls and uncertified candidates remain absent from the Kaggle runner
