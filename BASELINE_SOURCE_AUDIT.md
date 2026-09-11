# Revision baseline source audit

The final revision intentionally excludes local engineering variants from the
main literature comparison.  The paper-facing method names correspond to
published methods:

- Visual Prompting — Bahng et al.
- Conv-Adapter — Chen et al.
- Piggyback — Mallya et al.
- SSF — Lian et al.
- AdaptFormer — Chen et al.
- RepAdapter — Luo et al.
- ARC — Dong et al.
- VPT-Shallow / VPT-Deep — Jia et al.
- ConvPass — Jie and Deng.
- FacT-TT / FacT-TK — Jie and Deng; implementation reference: `JieShibo/PETL-ViT`.
- VQT — Tu et al.; implementation reference: `andytu28/VQT`.
- SPT-LoRA / SPT-Adapter — He et al.; implementation reference: `ziplab/SPT`.
- ML-Decoder — Ridnik et al.; implementation reference: `Alibaba-MIIL/ML_Decoder`.
- SegAdapter — Peng and Kameyama; paper-equation reimplementation.

`lora`, `bitfit`, `sidetune`, `norm`, `bias`, `last_block`, and
`convpass_attn` are deliberately not main-baseline rows in the revised paper.
`lora` remains available only for the reviewer-requested matched-budget rank
sweep.
