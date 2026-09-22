# G-CREST-TRSO Novelty and Evidence Audit

## Decision

The earlier V6 and CVEST candidates should not be used as the final proposal.

- V6 contained a dense-rescue path and could silently return the Linear checkpoint.
- The first universal/CVEST version removed those branches, but completed ablations showed that its partition-consistency weighting and finite-sample variance terms had only modest effects in the controlled cases. Its gain was therefore mainly attributable to a per-layer full-core geometry, which is not a sufficiently sharp novelty claim.
- G-CREST replaces independent per-layer capacity decisions with one global, budget-free allocation rule. This is the component that is active in the completed experiments: it changes the number of allocated modes, removes low-evidence tensors when appropriate, reduces parameters relative to the prior universal core, and preserves or improves controlled accuracy.

## Final contribution

G-CREST constructs one evidence value for every eligible layer-mode pair, normalizes all values into one model-wide distribution, and derives the total mode count as

\[
R=\left\lceil\sqrt{D_0D_1}\right\rceil,
\]

where \(D_0\) is numerical support and \(D_1\) is Shannon effective support. The globally strongest \(R\) modes determine both layer selection and per-layer rank. Full tangent cores are trained only in the allocated subspaces.

This provides one answer to three adapter design questions:

1. How many total modes should the model receive?
2. Which tensors should receive them?
3. What rank should each allocated tensor use?

No external rank or total parameter budget is supplied.

## Nearest prior work and claim boundary

The literature search covered work available through 29 July 2026. Absolute novelty cannot be proved by search, but the closest families are clear:

- **LoRA:** low-rank additive updates and exact merging.
- **AdaLoRA:** adaptive singular-component allocation under a prescribed budget.
- **EVA:** activation-SVD initialization and adaptive ranks under a rank budget.
- **LoRA-GA and GoRA:** gradient-driven initialization and adaptive rank assignment.
- **GaLore/Q-GaLore:** optimization in gradient low-rank subspaces.
- **GPS/GRFT and related selective tuning:** gradient-driven parameter, row or column selection.
- **LoRA-DA and other uncertainty-aware initializations:** sampling variance in data-aware adapter initialization.

G-CREST must **not** claim that SVD, gradient subspaces, adaptive rank, variance estimation, full cores or merging are new individually.

The defensible claim is narrower:

> A model-wide, budget-free PEFT allocation rule that derives the retained mode count and its tensor/mode distribution jointly from one partition-consistency-weighted calibration-gradient evidence distribution, followed by full spectral-core learning in the allocated subspaces.

The paper should use “to the best of our knowledge” and explicitly discuss the nearest methods above.

## Rejected designs

The following alternatives were implemented and tested during redesign but were not retained:

| Candidate | Reason rejected |
|---|---|
| Dense rescue / readiness routing | Violates the one-rule requirement and can adapt every tensor |
| Linear-checkpoint fallback | Hides failed adaptation and is not novel |
| Per-layer Shannon rank | Strong ViT control, but independent local allocation is too close to existing adaptive-rank work |
| Pure global Shannon allocation | Reduced parameters strongly but under-allocated the hard CNN control |
| Diagonal-only core | Lost useful cross-mode interactions |
| Grassmann-consensus scalar coordinates | Too restrictive on the hard CNN control |
| Compressed or frozen classifier | Large accuracy degradation |
| Functional KL anchoring | Reduced adaptation accuracy in controlled tests |
| Reliability shrinkage / norm saturation | More elegant regularization, but consistently reduced accuracy |
| Variance-only refinement | Completed ablations showed no controlled effect by itself |

The retained global geometric dimension is the parameter-free midpoint between the overly broad numerical support and the overly concentrated Shannon support.

## Executed controlled results

| Control | Seeds | Fixed reference | G-CREST | Mean change | Outcome |
|---|---:|---:|---:|---:|---|
| Standard TinyViT | 5 | 97.933% | **99.833%** | **+1.900 pt** | 5 wins, 0 ties, 0 losses |
| Standard TinyCNN | 5 | 100.000% | 100.000% | 0.000 pt | ceiling tie on every seed |
| Hard TinyCNN | 6 | 92.708% | **92.854%** | **+0.146 pt** | 2 wins, 2 ties, 2 losses |
| Hard TinyCNN, 30 epochs | 1 | 98.125% | **99.000%** | **+0.875 pt** | long-horizon win |

For the five-seed TinyViT control:

- one-sided exact Wilcoxon \(p=0.03125\);
- two-sided exact Wilcoxon \(p=0.0625\);
- bootstrap 95% interval for the mean gain: approximately \([0.60,3.30]\) points.

The hard-CNN six-seed interval includes zero. It is a weak positive diagnostic, not evidence of universal superiority.

## Comparison with the previous universal candidate

On the five TinyViT seeds:

- previous local universal mean: 99.667%;
- G-CREST mean: **99.833%**;
- previous mean tangent-core parameters: about 177;
- G-CREST mean tangent-core parameters: about **125**.

On the 30-epoch hard-CNN seed:

- fixed reference: 98.125%;
- previous local universal result: 98.375%;
- G-CREST: **99.000%**.

Thus the global geometric allocation improves the stress result and reduces capacity relative to the previous local rule.

## What is still unverified

No CPU synthetic experiment can establish publication-level performance on DTD, Flowers-102, Oxford-IIIT Pet, VOC2007 or segmentation. The next mandatory test is the corrected Session 1 DTD ResNet-50 run. A paper claim requires:

- a genuinely trained best epoch \(\ge 0\);
- no exact equality caused by Linear fallback;
- multi-seed uncertainty for the main comparison;
- ablations for global allocation, partition-consistency weighting, full-core interactions and the task-head policy;
- comparison against faithful architecture-compatible baselines.

The repository is ready for that test, but real-dataset superiority is not claimed in advance.
