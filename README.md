# Reliability-Weighted Spectral Allocation with Full Spectral Cores for Parameter-Efficient Visual Fine-Tuning

Official research implementation accompanying the manuscript:

**Reliability-Weighted Spectral Allocation with Full Spectral Cores for Parameter-Efficient Visual Fine-Tuning**  
Ba Ty Dang, Kim Huong Tran, Thi Uyen Nguyen  
**Manuscript submitted to _Pattern Analysis and Applications_.**

The internal implementation/CLI name remains **G-CREST-TRSO / `trso`** for backward compatibility with existing checkpoints, scripts, and the canonical 46-run protocol. Public documentation follows the manuscript terminology: the method uses **two deterministic calibration partitions for partition-consistency weighting**; their agreement is not statistical cross-fitting or repeated-run reproducibility.

## Active proposal

For every eligible CNN or Transformer weight, the method forms pooled calibration-gradient spectral modes and weights them by consistency between two deterministic calibration partitions. A single model-wide evidence distribution determines

\[
R=\left\lceil\sqrt{D_0D_1}\right\rceil,
\]

then allocates the globally strongest modes to tensors. Each participating tensor learns a full spectral core in \(U_{\ell,S_\ell}K_\ell V_{\ell,S_\ell}^\top\). The default proposal is intentionally unchanged from the manuscript.

The full proposal has:

- no user-specified global rank budget, evidence threshold, or layer list;
- the same allocation rule for evaluated convolutional and Transformer tensors;
- full spectral cores rather than diagonal-only scaling;
- exact algebraic merging, leaving **no active method-specific inference branch** after merging;
- a complete merged backbone checkpoint per downstream task.

See `METHOD_GCREST_TRSO.md` and the manuscript for the formal definition.

## Dataset and revision extension

The repository now exposes **54 canonical dataset routes** through a registry-backed interface. `--download auto` downloads only routes marked safe for unattended acquisition; `--download yes` explicitly permits large automatic routes, while manual/terms-sensitive datasets remain manual. Legacy `--download True/False` commands are still accepted.

Preprocessing is centralized and can resolve normalization/interpolation metadata from the pretrained checkpoint (`--preprocess auto`) while keeping all methods on the same backbone preprocessing contract. Optional MedMNIST routes include `pathmnist`, `dermamnist`, `bloodmnist`, `pneumoniamnist`, `organamnist`, and `tissuemnist`.

Canonical Kaggle Sessions **01-06 remain unchanged**. Sessions **07-14** are additional revision experiments for multi-seed robustness, reliability/allocation ablations, calibration sensitivity, and cross-domain generalization.

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
