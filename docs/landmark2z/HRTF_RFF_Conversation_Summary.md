# HRTF Landmark → Latent / RFF Experiment — Conversation Summary

## 1. Project context

Current experimental pipeline:

\[
	ext{3-D anatomical landmarks}
ightarrow
	ext{regression}
ightarrow
	ext{autoencoder latent } Z
ightarrow
	ext{frozen decoder}
ightarrow
\widehat{\mathrm{HRTF}}
\]

Important corrected context:

- Older landmark→latent experiments had incorrect regression subject numbering/mapping.
- An older landmark/anthropometric source had been omitted.
- Therefore older 52-subject / 33-train results are not treated as the final corrected evidence.
- Current corrected experiment uses **N=61 subjects**, with **46 development subjects** and **15 untouched test subjects**.
- Current geometry input uses **P=19 3-D landmark points**.
- Frequency range: **DC–15.85 kHz**.
- Latent PCA is now **fold-local** to avoid leakage.

Relevant notebooks:

- `FFM_2_8_Landmark2Z.ipynb`
- `FFM_2_9_RFF_MLP_Fixed.ipynb`

## 2. Why Random Fourier Features (RFF) were tested

Supervisor-requested idea:

\[
p_i=[x_i,y_i,z_i]^T
\]

\[
\gamma(p_i)=
[\sin(2\pi Bp_i),\cos(2\pi Bp_i)]
\]

with

\[
B\in\mathbb R^{M	imes3},
\qquad
B_{ij}\sim\mathcal N(0,\sigma^2).
\]

RFF is an **input encoder**, not the final predictor.

## 3. Ridge vs MLP

### Ridge

Regularized linear regression:

\[
\hat Y=XW+b
\]

with objective approximately

\[
\|Y-XW\|^2+lpha\|W\|^2.
\]

After RFF, Ridge is linear in Fourier features but nonlinear in the original XYZ coordinates.

### MLP

A nonlinear neural network. Current setup is approximately:

- 2 hidden layers
- 128 units
- GELU
- dropout 0.2
- AdamW

Current evidence shows MLP capacity is not the main bottleneck because it can nearly memorize training subjects.

## 4. Per-landmark RFF vs Global RFF

### Per-landmark RFF

Each landmark is encoded independently using the same \(B\in\mathbb R^{M	imes3}\):

\[
p_iightarrow\gamma(p_i)
\]

then all \(P\) landmark encodings are concatenated.

Output dimension:

\[
D=2PM.
\]

For the current experiment:

\[
P=19,\quad M=32
\]

so

\[
D=2	imes19	imes32=1216.
\]

### Global RFF

First flatten all landmarks:

\[
X=[x_1,y_1,z_1,\ldots,x_P,y_P,z_P]\in\mathbb R^{3P}
\]

then Fourier-encode the whole vector at once.

This mixes all landmarks together before the sine/cosine mapping.

## 5. Coordinate preprocessing

Current preprocessing:

\[
	ext{subject-centroid centering}
\]

For each subject:

\[
ar p=rac1P\sum_i p_i,
\qquad
p_i\leftarrow p_i-ar p.
\]

Coordinates are converted to metres.

No per-subject scale normalization is used because absolute head/ear size may contain useful signal.

## 6. Meaning of P, M and sigma

### P

\[
P=19
\]

is the number of anatomical landmark points per subject.

### M

\(M\) is the number of random Fourier vectors per landmark.

Each landmark gives \(M\) sine + \(M\) cosine features:

\[
D=2PM.
\]

With \(P=19\):

| M | RFF dimension |
|---:|---:|
| 4 | 152 |
| 8 | 304 |
| 16 | 608 |
| 32 | 1216 |
| 512 | 19456 |

### sigma

\[
B_{ij}\sim\mathcal N(0,\sigma^2).
\]

Larger sigma means faster spatial oscillation and sensitivity to finer coordinate changes.

Smaller sigma means smoother, more nearly linear features.

## 7. Cross-validation (CV)

CV = cross-validation.

It estimates performance on unseen subjects by repeatedly fitting on one subset of the development set and evaluating on a held-out fold.

Interpretation:

- \(R^2=1\): perfect prediction
- \(R^2=0\): same as predicting the training-fold target mean
- \(R^2<0\): worse than predicting the mean

## 8. Corrected main results

### Ridge

| Model | CV-R² |
|---|---:|
| Raw XYZ + Ridge | +0.0762 |
| Per-landmark RFF + Ridge | +0.0821 |
| Raw + Per-landmark RFF + Ridge | **+0.0845** |
| Global RFF + Ridge | +0.0038 |

Selected per-landmark configuration:

\[
M=32,\quad \sigma=20,\quad lpha=10^4.
\]

Raw→RFF Ridge gain:

\[
0.0821-0.0762=+0.0059.
\]

This difference is too small relative to CV variability to claim a meaningful improvement.

Best headline result:

\[
R^2=0.0845<0.10
\]

so the pre-specified stopping rule is triggered.

### MLP

| Model | CV-R² | train R² |
|---|---:|---:|
| Raw XYZ + MLP | -0.3330 | +0.8022 |
| Per-landmark RFF + MLP | -0.1395 | +0.9787 |
| Raw + Per-landmark RFF + MLP | -0.1310 | +0.9790 |
| Global RFF + MLP | -0.0147 | +0.9804 |

Interpretation:

RFF makes the MLP much easier to fit on training subjects, but this does not translate into unseen-subject generalization.

## 9. Corrected negative control

The fixed notebook now runs 20 target permutations with their own Ridge regularization selection.

Null distribution:

- median CV-R² ≈ **-0.0009**
- 95% interval ≈ **[-0.0026, +0.0104]**
- raw XYZ observed CV-R² = **+0.0762**
- empirical permutation p = **0.0476** with 20 permutations

Interpretation:

There appears to be detectable geometry→latent signal, but its predictive magnitude is small.

## 10. Extended sigma diagnostic

Extended grid:

\[
\sigma\in\{2,5,10,20,50,100,200,500,1000\}.
\]

Output:

| sigma | rough spatial scale | best CV-R² | best M |
|---:|---:|---:|---:|
| 2 | 500 mm | +0.0774 | 32 |
| 5 | 200 mm | **+0.0775** | 32 |
| 10 | 100 mm | +0.0759 | 32 |
| 20 | 50 mm | +0.0729 | 32 |
| 50 | 20 mm | +0.0602 | 8 |
| 100 | 10 mm | +0.0417 | 4 |
| 200 | 5 mm | +0.0103 | 4 |
| 500 | 2 mm | ~0.0000 | 4 |
| 1000 | 1 mm | ~0.0000 | 8 |

Extended-grid optimum:

\[
\sigma=5
\]

with coarse-scale / nearly-linear behavior.

Most importantly, performance collapses as the RFF is forced toward 1–5 mm scales.

## 11. Current interpretation

Current evidence suggests:

1. Geometry and latent are not independent; a weak detectable signal exists.
2. Raw XYZ + strongly regularized Ridge already captures most usable signal.
3. Per-landmark RFF gives at most a very small Ridge improvement.
4. Global RFF is ineffective.
5. RFF makes MLP memorization easier but does not improve unseen-subject prediction.
6. Fine-scale Fourier features (1–5 mm) are strongly disfavored by CV.
7. Sparse landmarks likely capture mainly coarse head/ear geometry rather than detailed pinna surface structure.
8. Increasing nonlinear capacity cannot create anatomical information absent from the landmark measurements.

## 12. Should M be increased to 512?

Current per-landmark grid:

\[
M\in\{4,8,16,32\}.
\]

The selected value \(M=32\) is at the upper boundary, so a **small M-extension diagnostic is defensible**.

However, jumping directly to \(M=512\) is unlikely to create a large improvement.

At \(M=512\):

\[
D=2PM=19,456.
\]

Reasons a large gain is unlikely:

1. The input becomes enormous relative to ~37 training subjects per fold.
2. MLP already reaches train \(R^2pprox0.98\), so representation capacity is not the bottleneck.
3. Fine-scale sigma values prefer smaller M (4 or 8), not larger M.
4. Increasing M mostly gives a denser Monte-Carlo approximation of the same RFF/kernel family.
5. It does not add new anatomical measurements.

If a diagnostic is required, a sensible sweep is:

\[
M\in\{32,64,128,256,512\}
\]

using **Ridge only**, preferably at coarse sigma values actually favored by CV, and with multiple RFF seeds.

Expectation: possible small stabilization or small gain, not a dramatic rescue.

## 13. Should sigma be forced to 1–5 mm?

Technically yes, but only as a **physics-motivated ablation**, not as the main model.

The current diagnostic has already approximately tested this:

- 5 mm → CV-R² ≈ +0.0103
- 2 mm → ≈ 0
- 1 mm → ≈ 0

So forcing the model to fine pinna-scale Fourier frequencies makes generalization substantially worse.

This does **not** mean millimetre-scale ear structure is acoustically irrelevant.

It means:

> The current 19 sparse landmarks do not contain enough millimetre-scale geometric information for fine-scale RFF encoding to become predictive of the latent.

Dense mesh/surface geometry would be a different experiment.

## 14. Recommended next action

If a final capacity diagnostic is wanted:

1. Keep corrected subject mapping and fold-local PCA.
2. Use Ridge, not MLP.
3. Test

\[
M=[32,64,128,256,512].
\]

4. Focus on coarse sigma values such as

\[
\sigma=[2,5,10,20].
\]

5. Repeat across several RFF random seeds.
6. Plot CV-R² versus M.
7. Stop when the curve plateaus.
8. Keep this diagnostic separate from the pre-specified headline result.
9. Do not touch the final 15-subject test set until upstream choices are frozen.

## 15. Key conceptual distinction

RFF can address a **representation problem**:

> Can the predictor represent a complicated nonlinear function of geometry?

It cannot solve an **information problem**:

> Do 19 sparse landmarks actually contain enough information about detailed pinna shape to predict the latent?

Current results indicate that the information limitation is likely the dominant issue.
