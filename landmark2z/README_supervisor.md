# landmark2z — sparse-landmark to HRTF-latent results note

This is an explanatory results note for the corrected `landmark2z` experiment.
It describes the data, the statistics, and their limits. It is not an
argument for a particular replacement geometry representation.

For the short project overview see [README.md](README.md). The reproducible
record is [RESULTS_RFF_2026-09-05.md](../docs/landmark2z/RESULTS_RFF_2026-09-05.md).

## Read the numbers first: the statistical quantities used here

The most important reported number is held-out cross-validated \(R^2\). For a
single fold, it is

\[
\boxed{
R^2=1-
\frac{\sum\lVert Y_{\mathrm{held\ out}}-\widehat{Y}\rVert^2}
     {\sum\lVert Y_{\mathrm{held\ out}}-\overline{Y}_{\mathrm{training}}\rVert^2}
}
\]

Here \(Y\) is the fold-local PCA representation of the autoencoder latent,
\(\widehat{Y}\) is the prediction from landmarks, and
\(\overline{Y}_{\mathrm{training}}\) is the mean target among the subjects
used to fit that fold. The numerator is the model's squared prediction error;
the denominator is the squared error from simply predicting the training-fold
mean for every held-out listener.

| Quantity | Formula or construction | What it answers | What it does **not** answer |
|---|---|---|---|
| CV-\(R^2\) | Formula above, averaged across development-set folds | Does landmark input predict held-out latent targets better than the mean-target baseline? | It is not an HRTF dB RMSE, not an AE reconstruction score, and not literally the proportion of all physical HRTF variability explained. |
| Unbiased \(\mathrm{dCor}^2_u\) | \(\widehat{\mathrm{dCor}}^2_u=\widehat{\mathrm{dCov}}^2_u(X,Y)/\sqrt{\widehat{\mathrm{dVar}}^2_u(X)\widehat{\mathrm{dVar}}^2_u(Y)}\), using U-centred pairwise-distance matrices | Is there evidence of any landmark–latent dependence, including nonlinear dependence? | It is not an out-of-subject prediction score or a variance-explained percentage. |
| Permutation \(p\) | \(p=(1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(1+B)\), for \(B\) random label permutations | Would a statistic at least this large be unusual if subject pairing were broken? | It does not state the practical usefulness or magnitude of a prediction. |
| Ridge fit | \(\widehat W=\arg\min_W\lVert Y-XW\rVert_F^2+\alpha\lVert W\rVert_F^2\) | What regularised linear landmark-to-latent map best fits a training fold? | A good training fit alone does not establish held-out performance. |

Two details prevent common misreadings:

1. \(R^2=0\) means equal to the training-fold mean predictor; \(R^2<0\)
   means worse. A small positive value can be statistically detectable yet
   still practically weak.
2. The unbiased dCor² estimator can be slightly negative in finite samples.
   That is ordinary estimation noise around its null expectation of zero, not
   “negative dependence.”

Where a table reports fold-based “±” values, these are internal CV dispersion
summaries (approximately \(s/\sqrt{K}\) for \(K\) reported folds). They are
not independent-sample confidence intervals because CV training sets overlap.

## 1. Research question and correction

The question is deliberately narrow:

> Given the sparse landmarks available for one listener, can we predict that
> listener's latent code from the frozen FFM26 HRTF autoencoder?

This is not direct HRTF-in-dB prediction. It is a regression from landmark
geometry to an internal learned representation. A weak score is therefore a
bound on this particular landmark protocol and target; it is not evidence that
anatomy or geometry is unimportant.

The historical `FFM_2_4_Landmark2Z.ipynb` and
`FFM_2_5_MLP_train.ipynb` used an incorrect correspondence between landmark and
HRTF subjects. Their 52-subject numerical results are invalid and retained
only for provenance. Every number below uses the corrected one-to-one SYMARE
subject identity.

## 2. What the data objects mean

There are 61 independent SYMARE listeners. The statistical unit is one
**subject**, not one ear.

| Quantity | Shape | Meaning |
|---|---:|---|
| `xyz_ear` | `(122, 19, 3)` | Landmark configurations for 61 subjects × two ears, represented in a common left-ear frame. |
| `xyz_sub` | `(61, 19, 3)` | Subject-level geometry: the two ear configurations are averaged. This is the regression input. |
| `Z_ear` | `(122, 5952)` | Frozen-autoencoder latent for each ear; \(5952=93\) frequency bins × 64 latent coefficients. |
| `Z_sub` | `(61, 5952)` | Subject-level target: the two ear latents are averaged. |

The 19 selected landmarks are 13 ipsilateral pinna points, 5 mid-sagittal
points, and 1 ipsilateral shoulder point. They originate from a `(61, 33, 3)`
array in millimetres; the other points are not in the configured input.
Right-ear configurations are mirrored before averaging, so every subject is
expressed in one anatomical frame. The observed left/right difference is kept
as a transformation check; it does not create 122 independent people.

`FFM_2_8_Landmark2Z.ipynb` exports the landmarks in their original common
frame and millimetres. `FFM_2_9_RFF_MLP.ipynb` centres each subject's 19
points at that subject's landmark centroid and converts to metres before
fitting. Centring removes global translation. There is **no per-subject scale
normalisation**, so head and ear size differences remain available to a model.

The target is produced by the four-database `FFM26_best.pth` autoencoder.
Its 93 frequency bins cover DC–15.85 kHz at 44.1 kHz sampling. `Z_sub` is a
learned latent representation, not a measured HRTF and not an HRTF dB error.

## 3. Partitions and the meaning of “untouched test”

| FFM26 partition | Subjects | Role in landmark regression |
|---|---:|---|
| AE training | 37 | Part of the development set |
| AE validation | 9 | Part of the development set |
| AE test | 15 | Reserved final landmark-regression test set |
| **Total** | **61** | Correctly matched subjects |

The landmark-regression development set has 46 subjects (37 + 9). All
Ridge/RFF/MLP choices and all reported cross-validation scores are made inside
these 46 subjects. The final 15 subjects have not been used for a final
landmark-regression score or for tuning.

One qualification is worth stating. The non-parametric dependence screen in
Section 4 uses all 61 subject pairs before regression. Thus “untouched” means
untouched by regression fitting, feature/model selection, and final regression
evaluation; it does not mean excluded from that descriptive all-subject
dependence screen.

## 4. Is there any geometry–latent dependence?

Before fitting a predictor, `FFM_2_8` calculates the **unbiased squared
distance correlation**, \(\mathrm{dCor}^2_u\), between subject-level
landmarks and subject-level latent.

Distance correlation is a dependence statistic: it can respond to nonlinear
as well as linear relationships, and is zero in expectation under
independence. Its unbiased finite-sample estimator can be slightly negative;
that is sampling variation around zero, not “negative dependence.”
Crucially, \(\mathrm{dCor}^2_u\) is **not** a percentage of variance explained
and is not an out-of-subject prediction score.

| Check, at subject level | \(\mathrm{dCor}^2_u\) | Permutation \(p\) | Interpretation |
|---|---:|---:|---|
| Landmarks vs latent | +0.0710 | 0.0020 | A dependence is detected by this test. |
| Landmarks vs shuffled latent | −0.0083 | 0.6090 | Null control behaves as expected. |
| Random features vs latent | −0.0309 | 0.9100 | Null control behaves as expected. |

The \(p\)-value uses 999 random permutations of subject labels:

\[
p=\frac{1+\#\{\text{permuted statistic}\geq\text{observed statistic}\}}
        {1+999}.
\]

Thus \(p=0.0020\) means the observed statistic exceeded almost every statistic
obtained after deliberately breaking the subject pairing. It is evidence
against “no dependence” under this permutation procedure. It does **not**
mean that 7.10% of latent variance is explained.

### Dependence by latent frequency band

| Frequency band | Bins | \(\mathrm{dCor}^2_u\) | Permutation \(p\) | Careful reading |
|---|---:|---:|---:|---|
| DC–0.5 kHz | 3 | +0.1399 | 0.005 | Only three bins and known low-frequency pathology; data-quality diagnostic, not an anatomical conclusion. |
| 0.5–6 kHz | 32 | +0.1282 | 0.005 | Detectable dependence in this broad band. |
| 6–12 kHz | 35 | +0.0631 | 0.015 | Weaker but detectable dependence in this analysis. |
| 12–15.85 kHz | 23 | +0.0725 | 0.005 | Weak detectable dependence. |

The band tests use 199 permutations, so \(p\) has resolution \(1/200=0.005\).
Several rows displaying 0.005 therefore do not have equal effect size or
equal evidential strength. These figures locate a dependence statistic; they
are not prediction accuracies and do not identify a causal anatomical source.

## 5. How held-out predictive performance is measured

`FFM_2_9_RFF_MLP.ipynb` evaluates prediction by cross-validation within the
46-subject development set. In each fold, latent PCA, coordinate
standardisation, and regression fitting use **only** that fold's training
subjects. The held-out fold is transformed with training-fold quantities. This
fold-local protocol avoids the PCA leakage present in the historical analysis.

The reported score is

\[
R^2=1-
\frac{\sum\lVert Y_{\mathrm{held\ out}}-\widehat{Y}\rVert^2}
     {\sum\lVert Y_{\mathrm{held\ out}}-\overline{Y}_{\mathrm{training}}\rVert^2},
\]

where \(Y\) is the fold-local PCA representation of the latent and
\(\overline{Y}_{\mathrm{training}}\) is the training-fold mean target.

| CV-\(R^2\) | Meaning |
|---:|---|
| \(1\) | Perfect held-out prediction. |
| \(0\) | No better than predicting the training-fold mean latent for everybody. |
| \(>0\) | Better than that baseline. |
| \(<0\) | Worse than that baseline. |

This is latent-space CV-\(R^2\), not HRTF reconstruction RMSE or dB LSD. It
should not be read literally as “the proportion of all physical HRTF variation
explained,” because the target is a fold-specific PCA approximation to the
frozen latent.

## 6. Main development-set results

| Input + predictor | Development CV-\(R^2\) | Training \(R^2\) | Reading |
|---|---:|---:|---|
| Raw XYZ + Ridge | +0.0762 | — | Small positive out-of-subject signal. |
| Per-landmark RFF + Ridge | +0.0821 | — | Numerically close to raw XYZ. |
| **Raw XYZ + per-landmark RFF + Ridge** | **+0.0845** | — | Largest primary development score. |
| Global RFF + Ridge | +0.0038 | — | Approximately the mean-target baseline. |
| Raw XYZ + MLP | −0.3330 | +0.8022 | Training fit without held-out generalisation. |
| Per-landmark RFF + MLP | −0.1395 | +0.9787 | Strong memorisation; no held-out benefit. |
| Raw + per-landmark RFF + MLP | −0.1310 | +0.9790 | Strong memorisation; no held-out benefit. |
| Global RFF + MLP | −0.0147 | +0.9804 | Near-baseline held-out result despite training fit. |

The largest primary result is \(+0.0845\), below the pre-specified development
stopping threshold \(0.10\). This threshold is a project decision rule, not a
universal scientific boundary; its purpose is to avoid indefinite capacity
tuning after a weak development result.

The notebook prints fold-based “±” values. They are useful as internal CV
dispersion, but are **not** independent-sample confidence intervals: folds
share most training subjects, and MLP seeds reuse the same 46 people. The
difference between Raw XYZ + Ridge (+0.0762) and per-landmark RFF + Ridge
(+0.0821) is too small to justify a claim that RFF materially improves
generalisation.

### Shuffled-target Ridge control

| Quantity | Value | Meaning |
|---|---:|---|
| Null median CV-\(R^2\) | −0.0009 | Correctly close to the mean-target baseline. |
| Central 95% of 20 null runs | [−0.0026, +0.0104] | Coarse reference distribution, not a precise confidence interval. |
| Observed Raw XYZ CV-\(R^2\) | +0.0762 | Above this shuffled-label sample. |
| Empirical permutation \(p\) | 0.0476 | With 20 permutations, this is the smallest attainable non-zero one-sided \(p\), \(1/21\). |

This control supports the claim that the small positive Raw-XYZ score is not
typical of this particular shuffled-label sample. With only 20 permutations,
the \(p=0.0476\) result is resolution-limited and should be described as
suggestive, not as a precise significance estimate. It still does not make a
small predictive magnitude useful.

## 7. What RFF, \(M\), \(\sigma\), and Ridge mean

For each landmark \(p_i=[x_i,y_i,z_i]^T\), a shared random matrix
\(B\in\mathbb{R}^{M\times3}\) produces

\[
\gamma(p_i)=[\sin(2\pi Bp_i),\cos(2\pi Bp_i)].
\]

The same map is applied to each landmark separately, then all 19 outputs are
concatenated. Pure per-landmark RFF has \(2PM\) features. Appending raw XYZ
gives \(P(2M+3)\) features. Thus the headline raw+RFF setting with
\(P=19,M=32\) has \(19(64+3)=1273\) features for 46 development subjects.

| Term | Meaning here |
|---|---|
| \(P=19\) | Number of selected landmarks. |
| \(M\) | Number of random frequencies per 3-D landmark; increases feature capacity. |
| \(\sigma\) | Sampling scale for \(B\); larger values make features vary more rapidly with coordinate changes. |
| Per-landmark RFF | Maps each 3-D landmark separately, preserving landmark identity. |
| Global RFF | Maps one flattened 57-coordinate vector; an ablation, not the same representation. |
| Ridge \(\alpha\) | L2 regularisation strength; larger values shrink coefficients more strongly. |

Coordinates are in metres when RFF is applied. The approximate spatial-scale
diagnostic is:

| \(\sigma\) | Approximate encoded scale | Best Ridge CV-\(R^2\) | Best \(M\) |
|---:|---:|---:|---:|
| 2–10 | 500–100 mm | +0.0759 to **+0.0775** | 32 |
| 20 | 50 mm | +0.0729 | 32 |
| 50–100 | 20–10 mm | +0.0602 to +0.0417 | 8–4 |
| 200 | 5 mm | +0.0103 | 4 |
| 500–1000 | 2–1 mm | approximately 0 | 4–8 |

For this input, smoother/coarser encodings outperformed strongly oscillatory
1–5 mm encodings. This says only that re-expressing these 19 points at a fine
Fourier scale did not help this regression. It neither measures nor rules out
information in surface detail not present in the landmarks.

## 8. Supplementary capacity diagnostics

These diagnostics address sensible follow-up questions—whether the MLP was
trained long enough, whether \(M=32\) supplied too few Fourier features, and
whether the development set is simply too small. They are supplementary to the
primary fold-local CV table in Section 6. They must not replace that table or
be presented as a final-test result.

### 8.1 Does 5000 epochs rescue the MLP?

The long-horizon experiment uses **Raw XYZ + per-landmark RFF** with \(M=32\),
\(\sigma=20\), \(D=1273\), two 128-unit GELU layers, dropout 0.2, and AdamW.
It records the same model on real targets and a matched shuffled-target
control. Training \(R^2\) describes fit to subjects seen during optimisation;
the displayed CV value describes the withheld subjects in this diagnostic.

| Epoch | Real train \(R^2\) | Real held-out \(R^2\) | Shuffled train \(R^2\) | Shuffled held-out \(R^2\) |
|---:|---:|---:|---:|---:|
| 50 | +0.5176 | −0.0642 | +0.5353 | −0.4205 |
| 100 | +0.7455 | −0.1946 | +0.7415 | −0.5236 |
| 200 | +0.9114 | −0.2031 | +0.9114 | −0.5436 |
| 500 | +0.9773 | −0.1445 | +0.9782 | −0.4178 |
| 1,000 | +0.9843 | −0.0821 | +0.9830 | −0.2978 |
| 2,000 | +0.9889 | −0.0265 | +0.9887 | −0.2060 |
| **5,000** | **+0.9895** | **+0.0078** | **+0.9880** | **−0.1080** |

How to read the table:

- The network can fit the real labels almost perfectly, but it can also fit
  shuffled labels almost as well. In a \(D>N\) problem, high training
  \(R^2\) is therefore a capacity/memorisation observation, not evidence of
  meaningful geometry-to-latent prediction.
- The real held-out score rises from negative values to only +0.0078 by the
  pre-specified 5000-epoch endpoint. This is essentially the mean-target
  baseline on the scale defined in Section 5, and far below the primary Ridge
  development score (+0.0845).
- The shuffled held-out score remains negative. The separation from the real
  curve indicates that the optimisation is not completely blind to signal, but
  it is not large enough to support useful held-out prediction.

This is a **diagnostic**, not a new independent uncertainty estimate: it uses
one RFF draw and a fixed internal holdout. Its defensible conclusion is narrow:
running the same MLP much longer does not turn the observed landmark input
into a strong predictor.

### 8.2 Does increasing the number of Fourier features help?

This Ridge-only sweep changes \(M\) while retaining Raw XYZ + per-landmark
RFF. Ridge is used so that the comparison is not confounded by stochastic
optimisation. \(D=P(2M+3)\), so increasing \(M\) increases the input dimension
very rapidly.

| \(M\) | Input dimension \(D\) | Development CV-\(R^2\) | Internal CV dispersion | Training \(R^2\) | Selected \(\alpha\) |
|---:|---:|---:|---:|---:|---:|
| 8 | 361 | +0.0624 | 0.0257 | +0.4010 | \(10^3\) |
| 16 | 665 | +0.0630 | 0.0098 | +0.1413 | \(10^4\) |
| 32 | 1,273 | +0.0845 | 0.0155 | +0.2270 | \(10^4\) |
| 64 | 2,489 | +0.0878 | 0.0198 | +0.3458 | \(10^4\) |
| 128 | 4,921 | +0.0556 | 0.0263 | +0.4864 | \(10^4\) |
| 256 | 9,785 | +0.0815 | 0.0140 | +0.1923 | \(10^5\) |
| 512 | 19,513 | +0.0932 | 0.0188 | +0.3010 | \(10^5\) |

The maximum observed score is +0.0932 at \(M=512\), only +0.0087 above the
\(M=32\) primary setting. The trend is not monotonic: the score falls at
\(M=128\), then rises again. The sweep therefore does **not** show a clear
capacity law or demonstrate that \(M=512\) is superior. The “internal CV
dispersion” column is not an independent standard error, for the same
overlapping-fold reason described in Section 6.

This sweep was conducted with one RFF seed. It is useful as a saturation check,
but a claim about the best \(M\) would require repeated RFF seeds and
pre-specified aggregation.

### 8.3 Is there evidence that more subjects could help?

The development-only learning curve holds out an **internal** 15-subject
evaluation set from the 46 development subjects. The remaining 31 subjects
form the pool from which training subsets are drawn. This is not the reserved
final 15-subject test set.

| Training subjects | Raw XYZ held-out \(R^2\) | Raw + RFF held-out \(R^2\) |
|---:|---:|---:|
| 8 | +0.0171 | +0.0079 |
| 12 | +0.0029 | −0.0140 |
| 16 | +0.0590 | +0.0538 |
| 20 | +0.0441 | +0.0511 |
| 24 | +0.0692 | +0.0790 |
| 28 | +0.0678 | +0.0772 |
| 31 | +0.0712 | +0.0819 |

For Raw + RFF, a straight-line fit over these seven points has slope
\(+0.00397\ R^2\) per added training subject. That line is a compact
description of these observations, **not** a forecast: the points are noisy,
the training subsets overlap, and there is no reason for linear scaling to
continue outside 8–31 subjects.

The appropriate interpretation is modest. The curve is not flat, so the
current results do not prove an absolute information ceiling. At the same time,
all observed scores remain small, so these 61 sparse-landmark observations do
not yet demonstrate useful individual latent prediction.

## 9. Bounded conclusion

The corrected analysis supports:

- The 19-landmark input and frozen latent are statistically dependent under the
  all-subject distance-correlation permutation test.
- In the 46-subject development set, the usable out-of-subject latent signal
  is small: the largest primary CV-\(R^2\) is +0.0845.
- Per-landmark RFF does not show a material generalisation gain over raw XYZ
  under this protocol.
- MLP training scores are not success metrics here; held-out scores indicate
  overfitting.
- The 15-subject landmark-regression test set remains reserved until the
  upstream protocol and reporting decision are frozen.

The analysis does **not** establish that geometry cannot predict HRTFs, that
fine anatomical detail is unimportant, that another architecture cannot help,
or that any alternative representation will necessarily succeed.

## 10. Reproducibility map

1. `FFM_2_8_Landmark2Z.ipynb` verifies the corrected subject identity,
   extracts frozen FFM26 latents, runs dependence diagnostics, and exports
   `landmark2z_v28_cache.npz`.
2. `FFM_2_9_RFF_MLP.ipynb` performs fold-local PCA, standardisation, RFF,
   Ridge, and MLP comparisons. Final test evaluation remains disabled by
   default.

The source data and `FFM26_best.pth` checkpoint are not committed to this
repository. The notebooks and linked results document are the provenance record
for the reported development-set analysis.
