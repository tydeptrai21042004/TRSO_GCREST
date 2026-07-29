# RepAdapter addition and BAM removal

## Completed changes

- Removed BAM from the tuning-module package, CLI, model construction, compatibility registry, strict-auto suite, tests, preflight, documentation, validation metadata, source manifest, and checksums.
- Added `repadapter`, implementing the RepBlock route from *Towards Efficient Visual Adaption via Structural Re-parameterization*.
- Enforced strict public comparison scope: **ViT-B/16 + single-label image classification**. Unsupported architectures and tasks fail before training.
- Inserted one RepAdapter branch before attention and one before the MLP in every ViT block.
- Frozen pretrained backbone parameters; only RepAdapter parameters and the task head train.
- Added exact structural folding into attention QKV and the first MLP projection for zero-wrapper inference.

## Canonical parameters

| Parameter | Default |
|---|---:|
| adapter dimension | 8 |
| groups | 2 |
| scale | 1.0 |
| dropout | 0.1 |
| merge for final inference | true |

## Verification

- Full repository tests: **136 passed**.
- Strict-baseline preflight: **all passed**.
- RepAdapter trainable-gradient check: **passed**.
- Exact structural merge: **4 branches folded in the tiny two-block verification model**.
- Maximum absolute output difference before/after merge: **0.0** in the preflight run.
- TRSO proposal core: unchanged.
