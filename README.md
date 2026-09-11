# G-CREST-TRSO: Budget-Free Global Evidence Allocation for Vision PEFT

This repository contains one active proposal, strict literature baselines, separate reference controls, and a 46-run/six-session Kaggle protocol.

## Active proposal

**G-CREST-TRSO** stands for **Global Cross-Fitted Reproducibility-Entropy Spectral Tangent Core**.

For every eligible CNN or Transformer weight, the method collects odd/even calibration gradients. All layer-mode evidence values are normalized together, not independently layer by layer. Their global geometric information dimension

\[
R=\left\lceil\sqrt{D_0D_1}\right\rceil
\]

defines one automatic model-wide mode budget. The globally strongest modes determine both which tensors are adapted and each tensor's rank. Every allocated tensor learns a full tangent core \(K_\ell\) in \(U_{\ell,S_\ell}K_\ell V_{\ell,S_\ell}^\top\).

The full proposal has:

- no manual rank or global parameter budget;
- no layer list or evidence threshold;
- no CNN/Transformer-specific path;
- no loss/readiness gate or rescue mode;
- no Linear-checkpoint fallback;
- exact merge and zero extra deployed parameters.

The task head follows one fixed policy and remains fully trainable. See `METHOD_GCREST_TRSO.md` and `GCREST_NOVELTY_AND_EVIDENCE_AUDIT.md`.

## Controlled evidence

The included deterministic transfer controls are diagnostics, not real-dataset publication claims.

| Control | Seeds | Reference accuracy | G-CREST accuracy | Mean change |
|---|---:|---:|---:|---:|
| Standard TinyViT | 5 | 97.933% | **99.833%** | **+1.900 pt** |
| Standard TinyCNN | 5 | 100.000% | 100.000% | 0.000 pt |
| Hard TinyCNN | 6 | 92.708% | **92.854%** | **+0.146 pt** |
| Hard TinyCNN, 30 epochs | 1 | 98.125% | **99.000%** | **+0.875 pt** |

G-CREST won all five standard ViT seeds. The hard-CNN six-seed improvement is small and not statistically conclusive; it is reported rather than hidden.

## Proposal ablations

```bash
python -m tools.run_trso_ablation \
  --dataset dtd \
  --data_path /path/to/data \
  --backbone resnet50 \
  --model_source torchvision \
  --seeds 0,1,2 \
  --epochs 30 \
  --execute
```

The analysis-only variants are `diagonal_only`, `no_sampling_variance`, `no_crossfit`, and `head_only`. They are not capacity knobs of the full proposal.

## Six Kaggle sessions

The paper protocol contains 46 training runs split into sessions of 6, 10, 12, 6, 6 and 6 runs. Every session:

- clones `https://github.com/tydeptrai21042004/trso_adapter.git` from `main`;
- verifies the G-CREST release contract;
- uses seed 0, 30 epochs and shared strong augmentation;
- separates strict baselines, reference controls, proposal rows and proposal ablations;
- records the resolved Git commit in `run_summary.json`.

See `kaggle/README.md` and `MINIMAL_PAPER_46_RUN_PROTOCOL.md`.

## Validation

```bash
pytest -q
python -m tools.validate_gcrest_controlled --suite standard --seeds 0,1,2,3,4
python -m tools.clean_release --check --manifest --zip ../TRSO_GCREST_release.zip
```

Repository tests and controlled diagnostics establish implementation correctness. A new real DTD ResNet-50 Session 1 run is required before claiming real-dataset superiority.

## Major-revision reviewer experiments

See [`REVIEWER_REVISION_GUIDE.md`](REVIEWER_REVISION_GUIDE.md) for multi-seed, allocation-stability, calibration-sensitivity, D0/D1-rule, R-scaling, and matched-budget LoRA experiments. The proposal defaults are unchanged; all new controls are ablation/evaluation switches.
