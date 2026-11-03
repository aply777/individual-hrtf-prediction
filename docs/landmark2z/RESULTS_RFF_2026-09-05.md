# Corrected landmark → latent / RFF experiment

**Status:** completed development-set study; the 15-subject test set remains untouched.

This is the current record for the sparse-landmark experiment. It supersedes every
quantitative conclusion from `FFM_2_4_Landmark2Z.ipynb` and
`FFM_2_5_MLP_train.ipynb`: those notebooks used an incorrect subject correspondence.
They remain in the repository for provenance only.

## Question and protocol

Can 19 three-dimensional anatomical landmarks predict an individual's frozen-autoencoder
latent? Coordinates are centred at each subject's landmark centroid, converted to metres,
and are not scale-normalised. The target is fitted with fold-local PCA and all input
standardisation and model selection are fold-local.

There are 61 correctly matched subjects: 46 form the development set and 15 are reserved
as an untouched final test set. Results below cover DC–15.85 kHz. The pre-specified stopping
rule is to stop capacity tuning if development cross-validated R² is below 0.10.

The workflow is implemented by
[`landmark2z/FFM_2_8_Landmark2Z.ipynb`](../../landmark2z/FFM_2_8_Landmark2Z.ipynb) and
[`landmark2z/FFM_2_9_RFF_MLP.ipynb`](../../landmark2z/FFM_2_9_RFF_MLP.ipynb). A fuller
working summary, including the explanatory discussion of RFF, is in
[`HRTF_RFF_Conversation_Summary.md`](HRTF_RFF_Conversation_Summary.md).

## Dependence check before fitting

The corrected subject-level dependence statistic is dCor²ᵤ = **0.0710** with permutation
p = **0.0020**. Negative controls are clean: shuffled latents give −0.0083 (p = 0.6090),
and random features give −0.0309 (p = 0.9100). Thus a geometry–latent relationship is
detectable, but this statistic establishes the presence of signal, not its predictive size.

| Frequency band | dCor²ᵤ | permutation p |
|---|---:|---:|
| DC–0.5 kHz | 0.1399 | 0.005 |
| 0.5–6 kHz | 0.1282 | 0.005 |
| 6–12 kHz | 0.0631 | 0.015 |
| 12–15.85 kHz | 0.0725 | 0.005 |

The DC–0.5 kHz row has only three bins and known data pathology; it is a data-quality
diagnostic rather than evidence about head geometry.

## RFF ablation

Each landmark \(p_i\in\mathbb{R}^3\) is encoded independently with a shared random matrix
\(B\in\mathbb{R}^{M\times3}\):

\[
\gamma(p_i)=[\sin(2\pi Bp_i),\cos(2\pi Bp_i)].
\]

The resulting per-landmark embeddings are concatenated, so the RFF dimension is
\(2PM\), where \(P=19\). This is distinct from the global-RFF ablation, which applies
one RFF map after flattening all 57 coordinates.

| Model | Development CV-R² |
|---|---:|
| Raw XYZ + Ridge | +0.0762 |
| Per-landmark RFF + Ridge | +0.0821 |
| Raw XYZ + per-landmark RFF + Ridge | **+0.0845** |
| Global RFF + Ridge | +0.0038 |
| Raw XYZ + MLP | −0.3330 (train +0.8022) |
| Per-landmark RFF + MLP | −0.1395 (train +0.9787) |
| Raw + per-landmark RFF + MLP | −0.1310 (train +0.9790) |
| Global RFF + MLP | −0.0147 (train +0.9804) |

The Ridge-only RFF gain is too small relative to cross-validation variability to support a
claim that RFF improves generalisation. MLPs nearly memorise the development subjects but
generalise below the mean-predictor baseline, so capacity is not the limiting factor.

A 20-permutation Ridge negative control has median CV-R² −0.0009 and 95% interval
[−0.0026, +0.0104]. The observed raw-XYZ score is +0.0762; its empirical permutation
p-value is 0.0476. The signal is therefore detectable but small in predictive magnitude.
The stopping rule is triggered by the best headline score, +0.0845 < 0.10.

## Fourier scale diagnostic

The original RFF search selected \(M=32,\sigma=20\), but \(\sigma=20\) was the lower
edge of its grid. Extending the scale grid showed that CV prefers an even smoother,
near-linear encoding:

| σ | Approximate spatial scale | Best CV-R² | Best M |
|---:|---:|---:|---:|
| 2 | 500 mm | +0.0774 | 32 |
| 5 | 200 mm | **+0.0775** | 32 |
| 10 | 100 mm | +0.0759 | 32 |
| 20 | 50 mm | +0.0729 | 32 |
| 50 | 20 mm | +0.0602 | 8 |
| 100 | 10 mm | +0.0417 | 4 |
| 200 | 5 mm | +0.0103 | 4 |
| 500 | 2 mm | ≈0 | 4 |
| 1000 | 1 mm | ≈0 | 8 |

This does not say that millimetre-scale pinna structures are acoustically irrelevant. It
says that 19 sparse landmarks do not retain enough millimetre-scale surface detail for a
fine-scale RFF representation to predict the latent. RFF changes the representation of the
measurements; it cannot add anatomical information that was never measured.

## Consequence and bounded follow-up

Sparse landmarks contain a weak, statistically detectable geometry–latent signal, but the
corrected experiment does not support useful unseen-subject prediction. The result is a
predictability bound for this landmark protocol and sample size, not a claim about dense
surface meshes or geometry in general.

If one final capacity diagnostic is needed, run Ridge only with
\(M=[32,64,128,256,512]\), \(\sigma=[2,5,10,20]\), and several RFF seeds, then plot
CV-R² against \(M\). Treat it as a saturation check, keep it separate from the headline
result, and do not access the reserved 15-subject test set until all choices are frozen.
