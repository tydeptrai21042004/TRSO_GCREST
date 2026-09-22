# Reliability-Weighted Spectral Allocation with Full Spectral Cores

> **Implementation compatibility note.** The internal method/CLI name remains
> **G-CREST-TRSO / `trso`**. The historical ablation token `no_crossfit` is also
> retained for old manifests and checkpoints, but in the manuscript it means
> **without partition-consistency weighting**. The two calibration partitions are
> not statistical cross-fitting and their agreement is not repeated-run
> reproducibility.

## One proposal rule

For every eligible pretrained matrix tensor \(W_\ell\), divide the calibration
batches into two deterministic partitions and compute the corresponding mean
gradients \(G_\ell^{(0)}\) and \(G_\ell^{(1)}\). With
\(B_f\) batches in partition \(f\), the pooled mean gradient is

\[
\bar G_\ell=
\frac{B_0G_\ell^{(0)}+B_1G_\ell^{(1)}}{B_0+B_1}
=U_\ell\operatorname{diag}(\sigma_\ell)V_\ell^\top.
\]

For singular mode \(j\), define the partition coordinates

\[
a_{\ell j}=u_{\ell j}^\top G_\ell^{(0)}v_{\ell j},\qquad
b_{\ell j}=u_{\ell j}^\top G_\ell^{(1)}v_{\ell j},
\]

with common and discrepant responses

\[
s_{\ell j}=\frac{a_{\ell j}+b_{\ell j}}{2},\qquad
d_{\ell j}=\frac{a_{\ell j}-b_{\ell j}}{2}.
\]

A tensor-level estimate of the average coordinate-wise variance of the pooled
mean gradient is

\[
\widehat\nu_\ell=
\frac{\sum_t\lVert G_{\ell t}-\bar G_\ell\rVert_F^2}
{B(B-1)m_\ell n_\ell}.
\]

Each candidate mode receives the partition-consistency weight

\[
q_{\ell j}=\frac{s_{\ell j}^2}
{s_{\ell j}^2+d_{\ell j}^2+\widehat\nu_\ell+\varepsilon},
\]

and evidence

\[
e_{\ell j}=\sigma_{\ell j}^2q_{\ell j}.
\]

The weight \(q_{\ell j}\) measures agreement **within the two calibration
partitions only**. It is not an estimator of independent cross-fit performance
or repeated-run stability. In the manuscript ablations, its predictive effect is
reported as modest in the tested settings.

All tensor-mode evidence values are normalized into one model-wide distribution

\[
p_{\ell j}=\frac{e_{\ell j}}{\sum_{a,b}e_{ab}}.
\]

Let

\[
D_1=\exp\!\left(-\sum_{\ell,j}p_{\ell j}\log p_{\ell j}\right)
\]

be the Shannon effective support and let \(D_0\) be the positive support of the
evidence vector. The default retained-mode count is

\[
\boxed{R=\left\lceil\sqrt{D_0D_1}\right\rceil}.
\]

This geometric rule is a **parameter-free default intermediate scale** between
\(D_1\) and \(D_0\); it is not claimed to be an accuracy-optimal allocation
criterion.

The globally strongest \(R\) evidence modes are retained. If tensor \(\ell\)
receives index set \(S_\ell\), its update is

\[
\boxed{
W_\ell^{\mathrm{eff}}
=W_\ell+U_{\ell,S_\ell}K_\ell V_{\ell,S_\ell}^{\top}
}.
\]

The core \(K_\ell\) is full on the selected left/right subspaces, so it defines a
larger within-subspace update family than diagonal spectral scaling. The
manuscript does **not** claim that this larger family consistently gives better
matched-capacity predictive performance. The task head follows one fixed policy
and remains fully trainable. After training, each learned update is merged into
the original tensor.

## What is not a proposal hyperparameter

The full method exposes no manually specified adapter rank, global mode budget,
evidence threshold, or layer list. Optimizer, learning rate, batch size,
augmentation, epoch count, and calibration construction are experimental
protocol choices rather than externally supplied global adapter-capacity
budgets.

The realized capacity is nevertheless **not independent of calibration
construction**: the candidate spectral dimension uses
\(\rho_\ell=\min\{m_\ell,n_\ell,B_0,B_1\}\), so calibration fraction/batching can
change the admissible candidate space. The manuscript reports this explicitly as
a limitation.

## Novelty and evidence boundary

Low-rank updates, SVD, gradient-derived subspaces, adaptive ranks, full spectral
cores, and exact merging are not claimed as individually new. The defensible
contribution is the joint model-wide procedure that derives the retained mode
count and tensor-specific allocation from one calibration-gradient evidence
distribution without a separately supplied global rank budget, then learns full
spectral cores in the selected subspaces.

The main empirical advantage claimed in the revised manuscript is **automatic,
task-dependent allocation**, not consistently superior predictive performance.
The LoRA rank sweep is an explicit-rank accuracy--parameter reference, not a
comprehensive matched-budget comparison with adaptive-rank PEFT methods.
Tensor-level allocation stability across seeds/partition realizations remains an
analysis question unless the optional stability diagnostics are run on completed
per-run calibration outputs.
