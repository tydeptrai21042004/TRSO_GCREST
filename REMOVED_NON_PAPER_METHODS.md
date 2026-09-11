> **Historical protocol notice (September 2026):** this file documents the frozen submitted-manuscript/46-run reproduction. It is preserved for reproducibility, but the active reviewer-ready baseline comparison uses `tools.run_fair_suite` with fresh heads and `tools.run_paper_fair_pairs` for paper-recipe paired checks.

# Methods excluded from the 46-run Kaggle comparison

The following methods remain unscheduled in all six Kaggle sessions:

- FacT-TT, FacT-TK, VQT
- SPT-LoRA, SPT-Adapter
- LoRA, BitFit, Side-Tuning
- Norm-only, Bias-only, Last-block
- ConvPass-Attn

Full Fine-Tuning and Linear Probing are **not removed**. They are restored as corrected, separately labeled reference controls. TRSO remains the proposal and its four variants remain proposal ablations.
