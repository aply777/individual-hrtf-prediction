# `landmark2z/` — 3-D landmarks to HRTF latent

This subproject tests whether sparse anatomical landmarks can predict the latent code of a
frozen HRTF autoencoder. It does **not** claim that a sparse landmark set is equivalent to
a head-and-pinna surface mesh.

## Current result

The corrected experiment uses 61 correctly matched SYMARE subjects, with 46 development
subjects and 15 untouched test subjects. Nineteen 3-D landmarks are centred per subject and
used to predict the latent with fold-local preprocessing and PCA.

There is a statistically detectable geometry–latent association (dCor²ᵤ = 0.0710,
permutation p = 0.0020), but the usable predictive signal is small. The best development
cross-validated score is **R² = 0.0845**, below the pre-specified 0.10 stopping threshold.
RFF brings no interpretable generalisation gain over raw coordinates; MLPs nearly memorise
the development subjects and fail to generalise.

The full protocol, tables, RFF scale diagnostic and permitted interpretation are in
[`../docs/landmark2z/RESULTS_RFF_2026-09-05.md`](../docs/landmark2z/RESULTS_RFF_2026-09-05.md).
The corresponding working/conversation summary is
[`../docs/landmark2z/HRTF_RFF_Conversation_Summary.md`](../docs/landmark2z/HRTF_RFF_Conversation_Summary.md).

## Data analysis and interpretation

### 1. Is there any geometry–latent signal?

Before fitting a predictor, `2_8` evaluates unbiased distance correlation between the
centred landmark representation and the latent. Across 61 subjects, dCor²ᵤ = **0.0710**
with permutation p = **0.0020**. Two negative controls are null (shuffled latent:
−0.0083, p = 0.6090; random features: −0.0309, p = 0.9100), so the result is not a
pipeline artefact.

| Latent frequency band | dCor²ᵤ | p | Interpretation |
|---|---:|---:|---|
| DC–0.5 kHz | 0.1399 | 0.005 | Only three bins with known pathology; treat as a data-quality diagnostic |
| 0.5–6 kHz | 0.1282 | 0.005 | Coarse geometry / head-size-related signal is detectable |
| 6–12 kHz | 0.0631 | 0.015 | Weak signal in the pinna-notch band |
| 12–15.85 kHz | 0.0725 | 0.005 | Weak high-frequency signal |

Distance correlation answers **whether a relationship exists**; it does not measure whether
the relationship is large enough to predict a new person's latent accurately. That second
question needs held-out cross-validation.

### 2. Can the relationship predict unseen subjects?

`2_9` performs all model selection inside the 46-subject development set. In every CV fold,
PCA, coordinate standardisation and Ridge fitting use only that fold's training subjects.
The 15 final test subjects have not been used.

| Input encoding and predictor | Development CV-R² | Training R² where relevant |
|---|---:|---:|
| Raw XYZ + Ridge | +0.0762 | — |
| Per-landmark RFF + Ridge | +0.0821 | — |
| Raw XYZ + per-landmark RFF + Ridge | **+0.0845** | — |
| Global RFF + Ridge | +0.0038 | — |
| Raw XYZ + MLP | −0.3330 | +0.8022 |
| Per-landmark RFF + MLP | −0.1395 | +0.9787 |
| Raw + per-landmark RFF + MLP | −0.1310 | +0.9790 |
| Global RFF + MLP | −0.0147 | +0.9804 |

The best score is below the pre-specified stopping threshold (CV-R² < 0.10). The small
Ridge difference between raw and RFF inputs is not large enough relative to CV variability
to support an RFF-improvement claim. The MLP results are more revealing than disappointing:
near-perfect training fit combined with negative held-out R² is overfitting, not evidence
that the network needs more layers or epochs.

A 20-permutation Ridge control gives a null median CV-R² of −0.0009 (95% interval
[−0.0026, +0.0104]); raw XYZ's +0.0762 has empirical p = 0.0476. In short, the data contain
a weak detectable signal, but not a useful predictive one under this protocol.

### 3. What did the RFF scale sweep add?

The per-landmark RFF has \(P=19\) points and feature dimension \(2PM\). Extending the
Fourier-scale sweep gave the following best Ridge result at each scale:

| σ | Approximate scale | Best CV-R² | Best M |
|---:|---:|---:|---:|
| 2–10 | 500–100 mm | +0.0759 to **+0.0775** | 32 |
| 20 | 50 mm | +0.0729 | 32 |
| 50–100 | 20–10 mm | +0.0602 to +0.0417 | 8–4 |
| 200 | 5 mm | +0.0103 | 4 |
| 500–1000 | 2–1 mm | ≈0 | 4–8 |

CV prefers the coarsest, almost-linear encodings and degrades steadily when RFF is forced
to resolve 1–5 mm structure. This **does not** imply that fine pinna features are acoustically
unimportant. It means that 19 sparse landmark points do not preserve enough fine surface
geometry for a millimetre-scale encoding to predict the latent. RFF can change a model's
representation of measured coordinates; it cannot create the missing anatomy.

### Conclusion and scope

The defensible conclusion is a bound: with this 19-landmark protocol and 61-subject sample,
sparse geometry has weak but statistically detectable association with the HRTF latent and
does not yield useful unseen-subject prediction. This is not a negative result about dense
meshes, full surface geometry, or HRTFs above 15.85 kHz. A final Ridge-only saturation check
over \(M=[32,64,128,256,512]\) and several RFF seeds is possible, but it must be kept
separate from the headline result and must not use the untouched test set during tuning.

## Run order

1. `FFM_2_8_Landmark2Z.ipynb` builds the correctly matched subject set, extracts the
   frozen-autoencoder latent, runs the non-parametric dependence diagnostics, and exports
   the cache for the next notebook.
2. `FFM_2_9_RFF_MLP.ipynb` runs fold-local Ridge and MLP comparisons for raw coordinates,
   per-landmark RFF and global RFF. It also runs the permutation and extended-σ diagnostics.

The notebooks require the four-database `FFM26_best.pth` checkpoint and the SYMARE data on
Drive; neither data nor weights are committed. All model selection is confined to the
46-subject development set. Do not run or use the 15-subject test set while tuning.

## RFF in this experiment

For each landmark \(p_i=[x_i,y_i,z_i]^T\), the same random matrix
\(B\in\mathbb{R}^{M\times3}\) produces
\(\gamma(p_i)=[\sin(2\pi Bp_i),\cos(2\pi Bp_i)]\). Concatenating the 19 embeddings gives
\(2PM\) features. This is an input representation for Ridge or an MLP, not a predictor in
its own right.

The extended scale sweep prefers coarse, almost-linear Fourier encodings and collapses at
1–5 mm scales. The appropriate interpretation is that the sparse landmarks lack the
surface detail required for that representation—not that fine pinna structure is
unimportant acoustically.

## Historical notebooks

`FFM_2_4_Landmark2Z.ipynb` and `FFM_2_5_MLP_train.ipynb` are retained only as a record of
the earlier analysis. Their 52-subject correspondence was wrong, so none of their numeric
results may be used. The corrected notebooks above supersede them.
