# Release contents

- `models/tuning_modules/mdl_tangent_core.py`: explicit G-CREST-TRSO proposal.
- `models/tuning_modules/arc.py`: strict ARC baseline for plain ViTs.
- `ARC_BASELINE_IMPLEMENTATION.md`: ARC mechanism, scope, defaults, and validation.
- `tools/minimal_paper_protocol.py`: canonical 46-run protocol and six-session allocation.
- `kaggle/TRSO_Paper_46_Session_01_OneCell.py` through `TRSO_Paper_46_Session_06_OneCell.py`: six self-contained training sessions.
- `kaggle/TRSO_Merge_6_Session_Results.py`: non-training result merger.
- `tools/run_trso_ablation.py`: separate TRSO ablation runner used in Session 3.
- `BASELINE_AND_ABLATION_SEPARATION.md`: publication taxonomy.
- `TRSO_Baselines_Reference_Controls_and_Ablations.docx`: updated experiment report.

Full Fine-Tuning and Linear Probing are restored as corrected reference controls. The Kaggle protocol still excludes transferred controls, engineering controls, paper-internal ablations, and uncertified reproduction candidates.
