# G-CREST-TRSO: Global Cross-Fitted Reproducibility-Entropy Spectral Tangent Core

## One proposal rule

For every eligible pretrained matrix tensor \(W_\ell\), split the complete calibration loader deterministically into odd and even batches and compute the corresponding mean gradients \(G_\ell^{(0)}\) and \(G_\ell^{(1)}\). The pooled response is

\[
\bar G_\ell=\tfrac12\bigl(G_\ell^{(0)}+G_\ell^{(1)}\bigr)
=U_\ell\operatorname{diag}(\sigma_\ell)V_\ell^\top.
\]

For singular mode \(j\), define the fold coordinates

\[
a_{\ell j}=u_{\ell j}^\top G_\ell^{(0)}v_{\ell j},\qquad
b_{\ell j}=u_{\ell j}^\top G_\ell^{(1)}v_{\ell j}.
\]

The finite-sample variance of a mean-gradient coordinate is estimated from the complete batch-gradient history as

\[
\widehat\nu_\ell=
\frac{\sum_t\lVert g_{\ell t}-\bar G_\ell\rVert_F^2}
{B(B-1)d_\ell}.
\]

Each mode receives one continuous reproducibility score

\[
q_{\ell j}=
\frac{[(a_{\ell j}+b_{\ell j})/2]^2}
{[(a_{\ell j}+b_{\ell j})/2]^2+[(a_{\ell j}-b_{\ell j})/2]^2+
\widehat\nu_\ell+\varepsilon},
\]

and evidence

\[
e_{\ell j}=\sigma_{\ell j}^2q_{\ell j}.
\]

Unlike layer-wise adaptive-rank methods, G-CREST pools **all layer-mode evidence values** into one model-wide distribution

\[
p_{\ell j}=\frac{e_{\ell j}}{\sum_{a,b}e_{ab}}.
\]

Let

\[
D_1=\exp\!\left(-\sum_{\ell,j}p_{\ell j}\log p_{\ell j}\right)
\]

be the Shannon effective support and let \(D_0\) be the numerical support of the evidence vector. The complete model-wide mode budget is not supplied by the user; it is

\[
\boxed{R=\left\lceil\sqrt{D_0D_1}\right\rceil}.
\]

The globally strongest \(R\) evidence modes are retained. If layer \(\ell\) receives index set \(S_\ell\), its update is

\[
\boxed{
W_\ell^{\mathrm{eff}}
=W_\ell+U_{\ell,S_\ell}K_\ell V_{\ell,S_\ell}^{\top}
}.
\]

The core \(K_\ell\) is full on the allocated subspace, allowing interactions among retained modes. A layer may receive no core when the global evidence allocation assigns it no mode. The classifier follows one fixed policy and remains fully trainable. After training, every core is exactly merged into the original tensor.

## What is not a proposal hyperparameter

The full method exposes no manual adapter rank, total parameter budget, layer list, energy threshold, evidence threshold, core type, calibration length, CNN/Transformer branch, loss/readiness gate, rescue route, or Linear-checkpoint fallback.

Optimizer, learning rate, batch size, augmentation and epoch count are shared training-protocol settings, not adapter-capacity controls.

## Novelty boundary

Low-rank updates, SVD, gradient-derived subspaces, adaptive ranks and exact merging are not claimed as individually new. The defensible contribution is the **joint model-wide rule** that derives both total adapter dimension and layer/mode allocation from one cross-fitted evidence distribution, without a prescribed rank or global budget, and then learns full tangent cores in the allocated subspaces.

A literature search cannot prove absolute novelty. The paper must compare explicitly with AdaLoRA, EVA, LoRA-GA/GoRA, GaLore-style gradient subspaces, and recent gradient-selection methods.
