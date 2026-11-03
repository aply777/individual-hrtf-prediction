# ELEC5021 Capstone Project B — Project Log

> **Public version (2026-11-03).** Omitted from the working log: the supervisor meeting notes (§3, §3A), the draft
> statement of personal contribution (§6), private correspondence with other authors, the details of the laboratory's
> unpublished method specification for Thread A, and every participant name. Where a passage was shortened, the
> result it supports is still stated. The numbering of the corrections register (§5–§5C) is unchanged, so references
> such as "log #16" in the thesis still apply.

| | |
|---|---|
| **Student** | Harry (Liaoyuan Peng) |
| **Unit** | ELEC5021 Capstone Project B, Semester 2, 2026 |
| **Supervisor** | Prof. Craig Jin |
| **Project** | Individualised HRTF prediction from 3-D geometry |
| **Log started** | 2026-08-08 |
| **Last updated** | 2026-10-01 |

---

## 0. How to use this log

Append a dated entry each working session. Each entry records: **what was attempted, what
the result was, what was concluded, and what is next.** Negative results and corrected
mistakes are recorded in full — they are evidence of process, not failures to hide.

Sections:

1. Project overview and current scope
2. Dated work log
4. Experiment index (script → question → result)
5. Corrections register (errors found in my own analysis and how they were caught)
7. Statement on the use of AI tools *(draft — for thesis)*

The log is append-only: later blocks (§2A–§2C, §4A–§4C, §5A–§5C, §8–§8C, §9B–§9C) continue
the numbered sections above rather than rewriting them. Where two blocks disagree, the later
one wins.

### 0A. Timeline at a glance

| Date | Entry | Outcome in one line |
|---|---|---|
| 08-08 | BEM reference frame + prior-art review | BEM vs measured set as the reference frame for Thread A |
| 08-15 | New data; FFM architecture review | Supervisor's additional CAR Lab subjects received |
| 08-17 | Latent capacity and structure; CAR Lab archive audit and export | 218 distinct CAR Lab subjects recovered from a reverse-engineered binary format |
| 08-21 | CAR Lab bandwidth | The 16 kHz ceiling is a filter, not the measurement |
| 08-22 | Model v2.3 | Three databases, 713 subjects, FFM + FiLM only |
| 08-23 | Diagnostics | Within DC–15.85 kHz the autoencoder is not the bottleneck |
| 08-24 | Training time; interruptible training | OSLOConv backward is 92% of an epoch; exact schedule continuity |
| 08-25 | Duplicate audit, four databases | HUTUBS repeat pairs found; no train/test leakage |
| 08-26 | SYMARE → CAR Lab mapping | *Retired 09-04: the cohorts do not overlap* |
| 08-28 | Per-subject error reporting | Test 1.4867 dB pooled / 1.2909 dB CAR Lab (v2.3) |
| 08-29 | Repository published; provenance audited | All four databases reproducible from the repository |
| 08-31 | Prior art read; landmark data reverse-engineered | 33 landmarks × 61 subjects; laterality settled acoustically |
| 09-01–03 | Geometry → latent, first pass | *Invalidated 09-04 with the mapping* |
| 09-04 | Cohort question answered; four-database retrain | FFM26: 774 subjects, val 1.4855 dB |
| 09-05 | Corrected landmark → latent (N = 61) | Detectable (p = 0.002), not usable (CV-R² 0.0845) |
| 09-06 | Capacity diagnostics | Longer training and more Fourier features do not rescue it |
| 09-07 | Revised Thread A specification studied | Mesh2HRTF/NumCalc platform; P0–P5 gates; thesis-student role |
| 09-09–11 | Supervisor results note; release staging | `landmark2z/README_supervisor.md` |
| 09-25 | Data-pipeline audit; literature survey | The "first subject's grid" concern verified harmless; two real issues found |
| 09-30 | NumCalc and Martinez et al. read | Thread A positioned against the nearest ICASSP 2026 work |
| 10-01 | Mesh → latent on 200 SONICOM meshes | Detectable (p = 0.010), not usable (6.5% of the gap); arm closed |
| 10-02 | Exploratory: end-to-end PointNet; LSD added | Learned point features lose to random features at N = 164 |
| 10-02 | Physics ceiling | An exact BEM solve from a person's own mesh predicts their measured HRTF worse than the mean — on SONICOM (optical) and SYMARE (MRI) alike |
| 10-04 | Within-person test | The left–right pinna difference predicts the left–right latent difference, but explains < 0.5% of it |
| 10-05 | Closing experiment: pinna only | Pinna-only crops and pinna alignment do worse; shape neighbours are only weakly z neighbours. **mesh → latent closed** |
| 10-05 | Fixed train / val / test split, epoch-trained MLPs | Same answer without cross-validation: MLP worse than the mean on both tasks; ridge 2.4% (meshes) |
| 10-06 | Fidelity audit of the mesh inputs; complete meshes packed; FFM_3_6 | No conversion error; 20k-point sampling noise (12–23%) and FFM_3_5's 64-PC MLP target flagged; exact per-triangle features ready to run |
| 10-06 | Complete-mesh results (FFM_3_6, FFM_3_7) | Exact = sampled within 0.003 dB; ridge 4.3% of the gap; MLP ≈ mean; whole-mesh graph network 3.7% at its best epoch, then overfits. Stopping rule triggered again |
| 10-07 | Literature check; proposed positioning (tentative) | No paper states the result; the pieces exist with incompatible baselines. Proposed: a baseline-referenced measurement answering Gaps 2–4 — not yet agreed |
| 10-07 | FFM26 held-out test, once | 1.46 dB pooled (CAR Lab 1.25) vs mean 4.81 (4.77); test ≈ validation. Test set spent |
| 10-10 | 48 kHz data pipeline (`preprocess_48k/`) | All six datasets at 48 kHz, every subject checked; found and fixed: CAR Lab grid-cache bug (CL154, CL159), ears reversed (CL003, CL088, SONICOM P0339), SONICOM P0209 excluded, CL169 duplicate directions |
| 10-10 | FFM48k_meas trained; simulation check | Val 1.4793 dB pooled (CAR Lab 1.358, HUTUBS 1.818, SONICOM 1.445, SYMARE 1.321). SONICOM simulations pass the 1.25 rule (0.93), SYMARE BEM fails (1.40) → FFM48k_measim trained for Task 1 |
| 10-11 | Task 1: geometry → simulated vs measured, same inputs and folds | SONICOM CV-R² 0.197 vs 0.069 (difference CI excludes 0); by band: 0.2–5 kHz sim 0.50 vs meas 0.12 (cause B), 6–16 kHz sim ≤ 0.23 (cause A); own simulation → measured R² 0.09 |
| 10-11 | Task 1b: torso added to the simulation (SYMARE HTE vs HE, 0.2–5 kHz) | HTE predicts measured slightly worse (R² 0.087 vs 0.104, CI [−0.036, −0.002]); the missing torso does not explain the low-band gap |
| 10-12 | Task 2: shape–HRTF modes, simulated vs measured | All six modes survive (cosine 0.27–0.73, p ≤ 0.012) at 0.79–1.11 × the simulated amplitude; measured low band has ~3 × the simulated between-person variance |
| 10-12 | Task 2 figures redrawn; variance table made reproducible | Script reproduces every value; ratio at 0.2–5 kHz 3.1 × [2.9, 3.4] (SONICOM), 3.5 × [3.0, 4.1] (SYMARE) |
| 10-12 | Autoencoder LSD added (FFM26 val + test, FFM48k val) | Recorded RMSEs reproduced (≤ 0.0001 dB). LSD: FFM26 test 1.299 vs mean 4.604; FFM48k_meas val 1.310 vs 4.770; FFM48k test not read |
| 10-12 | mesh → latent redone at 48 kHz, scored as S2RNF (development); test registered | LSD (bins 1–85, el ≥ −45°) below the mean by 4.0% (geometry), 5.3% (simulation), 5.8% (both); S2RNF 3.0% on its test. Not like for like; locked 36 not yet read |
| 10-12 | FFM_3_8 locked test (36 SONICOM), once | Primary confirmed: geometry 3.1% below the mean in LSD (−0.134 dB [−0.204, −0.068]); simulation 6.4%, both 6.4%. S2RNF 3.0% on its own test. Locked 36 spent |
| 10-15 | Why HUTUBS has the largest AE error | Only HUTUBS is measured below −45°; there the AE errs 2.3–2.4 dB LSD (others, extrapolated: 0.56–0.68). This accounts for about half of HUTUBS's excess; above −45° it is still the hardest |
| 10-16 | Decoder fine-tuning (supervisor's suggestion): design registered | Expected small and mostly not from geometry; primary statistic = geometry contribution with vs without fine-tuning, against a fine-tuned constant-input control; development-only (FFM_3_9, not yet written) |
| 10-16 | FFM_3_9: decoder fine-tuned on the prediction task (FiLM row / all biases / whole decoder) | The frozen decoder is not the limit: Δ −0.016 dB (A), below the 0.046 dB threshold; the geometry prediction itself moves ≤ 0.006 dB |
| 10-17 | Thesis: published-work comparison table; Chapter 2 split into background and literature review; outstanding revisions listed | `tab:sota` (all numbers checked in the papers); GISNO's 2.92 dB target not stated; 15 open items below |
| 10-23 | FFM_3_10 (head mesh outside the ear crops added to the geometry): design registered | Primary: full-band ΔLSD of GEO + w·REST against GEO, against a 100-permutation null; low-band ΔR² and the measured / simulated ratio (predicted ≈ 0.32) secondary; development only |
| 10-23 | FFM_3_10 result | No improvement: ΔLSD +0.003 dB [−0.004, +0.009]. Low band: R² +0.089 for the simulated latent but +0.008 for the measured one (ratio 0.08, predicted 0.32) — strengthens cause B |
| 10-27 | GISNO: reply from the first author; baseline check | Targets are measured HUTUBS HRIRs; LSD 0.2–8 kHz, both ears. Their training-mean baseline (6.252 dB) not reproduced: same protocol gives 3.11 dB [2.90, 3.43]; ≈ 6 dB only with complex averaging (a doubled dB scale contradicts the paper's Eq. 11). Thesis gives the mean, no reduction |
| 10-27 | Ch3 pipeline figure: panels for SONICOM P0010 (`FFM48k_PipelineFigure_Arrays`) | Gate passed: P0010 LSD of the decoded out-of-fold geometry prediction 3.8011 (recorded 3.8011), fold mean 3.8060 (3.8060); panels drawn locally, prompt for GPT written |
| 10-28 | SONICOM scan processing and hair: literature check | Hair-removal method and accuracy not reported; subjects wore a wig cap for both scan and HRTF measurement; behind-ear gaps interpolated. Hair added as an untested cause-B candidate (Treeby et al. 2007) |

---

## 1. Project overview and current scope

### 1.1 Two threads

| Thread | Description | Status |
|---|---|---|
| **A — Physics-structured surrogate** | A learned surrogate of the boundary-element simulation, designed in the thesis (Section 3.10). | Not implemented in this repository |
| **B — Spherical autoencoder (carried over from Unit A)** | HEALPix-native OSLO spherical AE with per-frequency latent + Fourier-feature frequency conditioning (FFM). Downstream: geometry → latent. | Trained; under diagnostic evaluation |

### 1.2 Thread A

The design is recorded in the thesis, Section 3.10. Its details belong to an unpublished laboratory method specification and are not reproduced here.

### 1.3 Deadlines

- **ICASSP submission: 16 September 2026** (Thread A, proof-of-concept slice)
- Thesis: end of semester (13 weeks)

---

## 2. Dated work log

### 2026-08-08 — BEM reference frame + prior-art review

**Attempted.**

1. Quantify the gap between SYMARE's BEM-simulated and acoustically measured HRTFs, to
   establish what reference frame the project's claims can be made against.
2. Review the closest prior art after identifying a possible novelty collision.

**Method.** SYMARE `HRIRs/Acoustic` (measured, 61 subjects × 2 ears) vs `Bem/HE16kHz`
(head+ears, valid ≤16 kHz) and `Bem/HTE05kHz` (head+torso+ears, ≤5 kHz), on the shared
393-direction grid. Magnitude spectra (delay-invariant); a single scalar dB gain removed
per ear (the two sets differ by ≈8 dB in source-strength convention).
Script: `analysis/bem_vs_measured.py`.

**Results.**

| Band | LSD(BEM, measured), gain-aligned |
|---|---|
| 0.2–16 kHz | **6.28 dB** |
| 0.2–5 kHz | 3.02 dB |
| 6–12 kHz (notch band) | **6.35 dB** |
| 12–16 kHz | 7.70 dB |
| mean-HRTF baseline | 8.38 dB |

- In the L1 metric used by Zhang et al. (2021): **4.66 dB** (they report 7.23 dB on HUTUBS).
- **Torso ablation: adding the torso changes the ≤5 kHz error by only +0.08 dB.** My prior
  hypothesis — that HUTUBS's torso-less meshes explained their larger gap — is **not
  supported**.
- **Individual-variation correlation: $r(\text{dev}_\text{measured},\text{dev}_\text{BEM}) = +0.756$**
  (0.775 in the notch band), i.e. **BEM explains $R^2\approx0.54$ of the between-subject
  variance**. Compare: 33 sparse landmarks → latent gave $R^2\approx 0$.
- Subject correspondence `HA_n ↔ HB_n` **confirmed** (21/61 subjects' true pairing is the
  global argmin; chance = 1/61). Incidentally resolves an open item from the Unit A notes.

**Conclusions.**

1. Zhang et al.'s "BEM converges to the wrong distribution" **replicates on SYMARE**.
   ⇒ All Thread-A gate metrics (P0–P3) must be reported **against BEM**, with the reference
   frame stated explicitly. No claim of beating measured-HRTF SOTA.
2. The honest ceiling: a perfect BEM surrogate lands at ~5.7–6.3 dB vs measurement,
   against a mean-HRTF baseline of 8.38 dB — a real but modest ~2.1–2.7 dB gain.
3. $R^2=0.54$ is the strongest quantitative argument for the geometry-based direction so far.

**Prior-art review — Zhang, Wang & James, "Personalized HRTF Modeling Using DNN-Augmented
BEM", ICASSP 2021.** Read in full. Their method replaces the *bias term* of a spatial-PCA
decomposition with the subject's BEM simulation and learns the BEM-vs-measurement residual
from 8 anthropometric parameters. **Not a collision:** they treat BEM as a black box and use
only its far-field output as an additive prior; a full BEM solve is still required per
subject at inference. This project learns *inside* the solve. Must be cited with an explicit
differentiation sentence; also cite SPCA-DNN (Zhang 2020).

**Open item.** SYMARE's `readme.txt` states the shipped BEM (v0.9, 2014) was to be re-run with
Burton–Miller and 20 kHz meshes for v1.0 — **confirm with Craig whether a newer set exists**,
as the current data may understate BEM's capability.

**Also produced.** A from-scratch derivation of the boundary integral equation
(`notes/BIE_derivation_from_scratch.tex`, 13 pp) and a 45-question concept quiz
(`notes/BIE_quiz.tex`), for my own preparation on Thread A.

---

### 2026-08-15 — New data; FFM architecture review

- Received additional HRTF data from supervisor (`SYMARE/New_hrtfs/`). Index file
  `carlabdb.mat` contains **267 records / 248 unique subject names** from the CAR Lab's own
  measurement archive, plus `SymareHrirs` (61), `AnlHrtfRecordings` (~a dozen).
  HRIRs are stored in the lab's binary `hrirfilt*` format; extraction requires the MATLAB
  tooling shipped alongside. **Action: confirm how many are usable.**
- Reviewed `FFM/Epoch300/New_FFM_1_0.ipynb`. Clarified the architecture: the encoder
  processes each of the 129 frequency bins **independently**, so the latent is a
  **(129 × 64) matrix per ear = 8256 numbers**, not a 64-vector. This distinction matters —
  the "is 32 dimensions enough?" discussion concerns a different (multi-dataset) model.

---

### 2026-08-17 — Latent capacity and structure (Thread B diagnostics)

**Motivation.** Resolve a disagreement about latent dimensionality: whether ~32 dimensions
is too small for a population of hundreds of subjects, and whether reconstruction quality is
limited by latent capacity, by the encoder, or by sample size. Five experiments, all on the
existing trained checkpoint (`FFM_1_best_model.pth`) — **no retraining**.

#### Experiment 1 — Intrinsic dimensionality of the HRTF data itself (no network)

SVD of measured SYMARE HRTF magnitudes (122 ears × flattened spectra), reporting **held-out**
LSD vs rank rather than variance-explained (variance-explained systematically overstates
sufficiency, because notches are low-variance but perceptually critical).
Script: `analysis/hrtf_svd_rank.py`.

| rank | in-sample LSD | **held-out LSD** |
|---|---|---|
| mean baseline | 8.59 | 8.59 |
| 8 | 3.51 | 3.84 |
| 16 | 3.07 | 3.63 |
| 32 | 2.53 | 3.46 |
| 121 (all) | 0.00 | **3.26** |

Per-frequency effective rank: **4 dims below 5 kHz, 16 in 6–12 kHz, 24 in 12–16 kHz.**

**Conclusion.** Held-out error plateaus at ~3.3 dB even for an optimal *linear* basis. A fixed
linear basis cannot **translate** a notch (it can only add and subtract fixed patterns), and
individual variation is largely notch *position* moving. This is a data-derived, model-free
argument for the physics-structured approach, where notch frequency is computed exactly.

#### Experiment 2 — Amortization gap (is the encoder leaving accuracy on the table?)

Froze the decoder; optimised the latent directly per ear (test-time optimisation) from two
starting points. Script: `analysis/amortization_gap_v2.py`.

| Band | mean baseline | encoder | warm-opt | cold-opt |
|---|---|---|---|---|
| overall | 9.697 | 2.326 | **2.158** | 2.161 |
| notch 6–12 k | 9.371 | 2.029 | 1.866 | 1.870 |

- **Amortization gap = 0.168 dB** (7% relative). Small but real; uniform across bands.
- **Cold start (from the population-mean code, encoder unused) reaches the same solution**
  as warm start: $\cos(z_\text{warm},z_\text{cold})=0.999$, difference +0.003 dB.
  ⇒ the optimisation landscape is benign and **the encoder contributes no information
  beyond a better initialisation**.

**Conclusion.** Encoder-based and auto-decoder formulations are **equivalent for
reconstruction**. The choice must be made on other grounds (handling sparse/heterogeneous
direction sampling; ability to impose priors on the code), not on reconstruction quality.

#### Experiment 3 — What does the latent actually encode?

Variance decomposition over 96 training ears (48 subjects), with explicit side/subject labels.
Script: `analysis/latent_variance_decomp.py`.

| Component | Share of latent variance |
|---|---|
| **"which ear" (left/right)** | **71.7%** |
| "which person" (subject given side) | 18.6% |
| residual | 9.6% |

- PC1: 72.1% of variance, $|r|$ with side = **0.997**, separates L/R with **100% accuracy**.
- PC2: 4.3% of variance, $|r|$ with overall dB level = **0.960**.

**Conclusion — the most consequential finding of the day.** Three quarters of the latent's
capacity encodes two quantities that are known a priori (which ear; overall gain).
Only **18.6%** encodes the listener.

Effective dimensionality (90% variance): 22 including side → **59 with side removed** →
**33 using left ears only**. The "22" was inflated by the dominant nuisance direction.
Against a hard ceiling of 47 set by 48 subjects, **33/47 = 70%** ⇒ **the estimate is
sample-limited, not dimension-limited.**

**Actions identified.** (a) Mirror right ears across the median plane before encoding
(`analysis/healpix_mirror.py`, verifies the HEALPix reflection is an exact permutation);
(b) normalise per-ear level; (c) regress downstream onto the leading principal components of
the *side-removed* latent, after per-dimension standardisation (the marginal std spans a
factor of 422, so an unstandardised MSE/$R^2$ is dominated by a few high-variance dimensions).

#### Experiment 4 — Does frequency-dependent capacity allocation help?

Constrained the latent to the top $d_f$ principal directions per frequency bin and optimised
within that subspace — giving the *ceiling* for each allocation policy without retraining.
Script: `analysis/freq_budget_ceiling_v2.py`. Self-check: the unconstrained (64/bin) case
reproduces Experiment 2's ceiling (2.192 vs 2.158 ✓).

| Policy | overall | low | notch | high |
|---|---|---|---|---|
| uniform 8 | 3.469 | 1.491 | 3.405 | 4.119 |
| uniform 16 | 2.992 | 1.255 | 2.792 | 3.628 |
| uniform 32 | 2.475 | 1.048 | 2.182 | 3.051 |
| uniform 64 (unconstrained) | **2.192** | 0.949 | 1.901 | 2.712 |
| proportional (= uniform-16 budget) | 2.912 | 1.685 | 2.936 | 3.311 |

**Conclusions.**

1. **Total capacity is a first-order effect; allocation is second-order.**
   16→64 dims buys **0.80 dB**; reallocation at fixed budget buys **0.08 dB**.
2. Reallocation is approximately **zero-sum** — gains where dimensions are added, loses
   where they are removed, roughly in proportion.
3. **Design limitation of this run:** the "proportional" profile gave the notch band 13 dims,
   *fewer* than uniform-16's 16. It therefore did **not** test "give the notch band more"; it
   tested "give >12 kHz more". To be re-run with a notch-weighted profile if pursued.
4. The unconstrained ceiling is 2.192 dB — that is the joint limit of **decoder architecture
   + 64 dims/bin**, not of the encoder (which reaches 2.326, only 0.13 dB away).

#### Consolidated conclusions for Thread B

| Question | Answer |
|---|---|
| Is the latent dimension the bottleneck? | Not primarily — 72% of capacity is spent on a known bit. After fixing that, capacity is adequate; more dims still help (+0.80 dB from 16→64) but with diminishing returns. |
| Is the encoder the bottleneck? | No — 0.168 dB from its own ceiling. |
| Is the decoder the bottleneck? | Partly — 2.19 dB even unconstrained. |
| **Is sample size the bottleneck?** | **Yes, for anything downstream.** Effective dimensionality (33) is at 70% of the ceiling imposed by 48 subjects. |

⇒ **Extracting the CAR Lab subjects is higher leverage than any architectural change.**

**Next.** Return to Thread A (ICASSP): implement `projection.py` with the rigid-sphere
analytic test (gate P0). Request from the PhD student: Direct vs Indirect BEM formulation,
and one sample Coustyx surface-results export.

---

### 2026-08-17 (later) — CAR Lab archive: audit, format reverse-engineering, export

**Objective.** Determine how many usable subjects the supervisor's new data actually
contains, and produce them in the same format as the existing training set.

#### Audit

`carlabdb.mat` is an index, not data. Following the chain:

| | count |
|---|---|
| index entries | 267 |
| `hrirfilt` files that exist | 259 (missing 81, 85, 113, 136, 149, 150, 204, 208) |
| ≥300 directions and valid waveforms | 241 |
| **unique people** | **219** (218 after excluding the HATS dummy head) |
| of which already in SYMARE | 56 |
| **net new** | **≈163** |

Direction grids: 393 (214 files, identical to SYMARE's), 475 (23), 412 (1), 392 (3),
plus 18 sparse sets (205/141/47) excluded. Script: `analysis/audit_carlab.py`.

#### Binary format reverse-engineered

The reader (`get_filter`, `get_filter_pos` from `carlabtoolbox`) is **not in the
repository**. Recovered the format from file-size regularities plus byte inspection:

```
[int16] L = 256                     filter length
N records of 4 + 2*L*4 = 2052 bytes:
    [int16 az][int16 el]            degrees
    [float32 x L] ear0 = left
    [float32 x L] ear1 = right
N = (filesize - 2) // 2052
```

**Cross-validated against SYMARE** (`hrirfilt82` against the same subject's SYMARE HRIR):
azimuth must be negated, after which all 393 directions match to **0.00°**; magnitude-spectrum
correlation `hL↔ear0 = +0.969`, `hR↔ear1 = +0.971`, cross-pairings −0.17/−0.20 (ear order
confirmed). So the data is fully readable in Python; MATLAB is not required.

Sample rate resolved: `carlabdb`'s `srate = 12.5` is the **sampling period in microseconds**
(`fsRec = round(1e6/srate)` → raw `.imp` at 80 kHz); the `hrirfilt` files are already
resampled to **48 kHz / 256 samples**. Confirmed independently: ITD at az = 90° is 708 μs
at 48 kHz, matching the physiological maximum (~700 μs); 44.1 kHz would give 771 μs.

#### Provenance of the existing HA01–61

- `SymareHrirs/HRIR_<name>.mat` is **byte-identical** to `HRIRs/Acoustic/HA<nn>.mat`
  (max abs diff 0.00e+00). This gave a complete name ↔ HA-number map (61/61), used to
  resolve the SYMARE/CARLab overlap precisely.
- `ProcessHRIRs.m` reads `hrirfilt` but **outputs `HRIR_ANL*`**, not HA files.
  `ProcessHRIRsSymare.m` *reads* the HA files and converts downstream.
  ⇒ **The script that produced HA01–61 is not in the repository.** Exact reproduction
  of their processing is not possible.

#### How different is raw from processed, and why

Gain-aligned, 500 Hz–16 kHz, on 42 correctly matched subjects: **3.81 dB**, decomposing as

| component | RMS |
|---|---|
| shared direction-independent EQ | 0.49 dB (2% of variance) |
| per-subject frequency response | 1.37 dB |
| **direction-dependent residual** | **3.52 dB** |

So a single estimated correction filter recovers almost nothing — the difference is
overwhelmingly direction-dependent. Script: `analysis/calib_transfer.py`.

I hypothesised the direction-dependent part was uncorrected head misalignment, and ported
the head-alignment stage of `ProcessCARLab_Hrirs.m` to Python to test it
(`analysis/head_align.py`; excess delay via min-phase cross-correlation, then
multilateration for the ear positions, then re-interpolation onto the nominal grid).
Two necessary deviations from the MATLAB original are documented in the script.

**Result: the hypothesis is wrong.** The recovered geometry is
head-centre offset **0.7–1.2 mm**, rotations **±0.1°**, direction corrections **0.03–0.06°** —
i.e. essentially zero, and applying the correction changes the error by 0.00 dB.
**`hrirfilt` has already been head-aligned** — that is evidently what "FixedByPierre" means.
A separate check showed the ILD difference does not match the near-field-removal signature
either (300–800 Hz differs by only 0.82 dB, whereas near-field removal would be largest at
low frequency).

**Consequence — this is good news.** The raw archive carries no pose nuisance, so using it
directly is not a compromise. Decision: **use one raw pipeline for all 218 subjects**,
including the 56 that also exist in SYMARE, and do not mix in the HA files.
Restrict training to **500 Hz–16 kHz** (below 500 Hz the measurement is unreliable — which
is why SYMARE substitutes a sphere model there; above 16 kHz the raw data rolls off to −42 dB).

#### Export

`analysis/export_carlab_npy.py` → `analysis/finalize_carlab.py`:

```
SYMARE/CARLab_left/    CL001_left_full.npy  … CL218_left_full.npy   (129, 3072) float32
SYMARE/CARLab_right/   CL001_right_full.npy … CL218_right_full.npy
SYMARE/CARLab_subjects.txt / .json      ID → name / carlabdb index / n_dirs
SYMARE/healpix_mirror_idx.npy           left/right mirror permutation
```

Same format and same algorithm as the existing `spherical_spline.py` pipeline (FFT →
20 log10 → STPS onto HEALPix nside=16); only the loader differs. A separate STPS matrix is
built per direction grid. Verified: no NaN/Inf, per-subject means −20.4 … −11.0 dB.

The HEALPix mirror was verified to be an **exact bijection and an involution**
(`idx[idx] == identity`), so mirroring introduces no interpolation error.
Empirically `corr(left, right) = 0.813` → `corr(left, mirrored right) = 0.978`.

**Note on mirroring both ways.** Mirroring the left ears instead would give the *same*
dataset up to a fixed pixel permutation, so training a second model in the "right-ear frame"
is redundant. Including *both* frames as augmentation would be actively harmful — it
reintroduces the frame bit that mirroring exists to remove.

#### Model v2

`FFM_2_0_CARLab.ipynb` — four changes from v1, each traceable to a measurement above:
monaural samples with right-ear mirroring (218 → 436 samples); train/val split **by subject**
(the two ears of one person correlate 0.978 after mirroring, so a per-sample split would leak);
ReLU → GELU; and a per-dataset FiLM layer for future multi-dataset training.

**FiLM design note.** The FiLM modulation is **per-frequency only, never per-direction**.
That is 129 degrees of freedom per dataset — enough to absorb a loudspeaker/microphone
response, an equalisation convention or an overall level, but structurally incapable of
representing direction-dependent structure. Population anatomy therefore still has to be
explained by the latent. This is the substantive difference from per-dataset mean removal,
which subtracts a full 129 × 3072 map and can absorb genuine population structure along with
the measurement artefact.

**Caveat for the new baseline.** The new data is a different distribution (raw pipeline,
218 subjects vs 48). The v1 figures (2.14 / 2.33 dB) are **not comparable**; the notebook
computes a fresh mean-HRTF baseline on the new data for reference.

---

## 4. Experiment index

| Script | Question | Headline result |
|---|---|---|
| `analysis/bem_vs_measured.py` | How far is SYMARE's BEM from measurement? | 6.28 dB; but $R^2$=0.54 of individual variation |
| `analysis/hrtf_svd_rank.py` | Intrinsic dimensionality of HRTF data (model-free) | Held-out floor 3.26 dB; 4→24 dims low→high frequency |
| `analysis/amortization_gap_v2.py` | Is the encoder near-optimal? | Gap 0.168 dB; auto-decoder equivalent |
| `analysis/latent_variance_decomp.py` | What does the latent encode? | 71.7% = which ear; 18.6% = which person |
| `analysis/latent_diagnostics_v2.py` | Effective dimensionality | 33 of a 47 ceiling (left ears) |
| `analysis/freq_budget_ceiling_v2.py` | Does frequency-wise reallocation help? | +0.08 dB vs +0.80 dB from more total capacity |
| `analysis/audit_carlab.py` | How many usable subjects in the CAR Lab archive? | 219 unique (218 excl. dummy head); ≈163 net new |
| `analysis/calib_transfer.py` | Is raw-vs-processed a fixed filter or direction-dependent? | 3.81 dB total, only 0.49 dB fixable by a filter |
| `analysis/head_align.py` | Does `hrirfilt` need head-alignment correction? | **No — already aligned** (0.7 mm, ±0.1°) |
| `analysis/export_carlab_npy.py` + `finalize_carlab.py` | Produce training npy in the existing format | 218 subjects × 2 ears, `CL001`…`CL218` |
| `analysis/healpix_mirror.py` | Is the HEALPix left/right mirror exact? | Exact bijection and involution ✓ |

---

## 5. Corrections register

Errors found in my own analysis during this project, and how they were caught. Recorded
because the corrections changed conclusions.

| # | Error | How caught | Effect on conclusions |
|---|---|---|---|
| 1 | Test-time optimisation used Adam `lr=3e-3` while the latent's per-component magnitude was ≈81 — the optimiser moved 1.5% and effectively did nothing. | The reported movement `‖Δz‖/‖z‖ = 0.013` matched `steps × lr / magnitude` exactly. | All three conclusions ("gap is small", "capacity limit reached", "bad local minima") were artefacts. Rerun after re-parameterising in units of the per-dimension std. |
| 2 | Same class of error in the PCA-coefficient parameterisation (coefficient scale up to 435, `lr=0.1`). | The unconstrained (64-dim) case returned 8.9 dB where it must reproduce the known 2.16 dB ceiling. | "Reallocation gives no benefit" was invalid. A **built-in self-check against a known answer** was added so this cannot pass silently again. |
| 3 | "Variance concentration" computed by sorting *marginal* per-dimension variances, which does not measure redundancy. Reported "4799 dimensions needed" for a matrix whose rank cannot exceed 95. | Rank bound $N-1=95$ contradicted the reported number. | Replaced with a proper SVD; the true figure is 22 (and 33 after removing the side direction). |
| 4 | Left/right label mask built as `arange % 2 == 0`, but the extraction loop appends a whole batch of left ears then a whole batch of right ears. | Classification accuracy came out at **exactly** 0.500 — too exact to be a measurement. | The left/right hypothesis had been wrongly rejected; with correct labels it is confirmed at 100% accuracy and 71.7% of variance. |
| 5 | Surname matching used `sur.rstrip("a…z")`, which strips *every* trailing letter — a surname became `""`, and `"" in k` is always true, so every SYMARE subject matched the first CARLab entry. | 61/61 subjects "matched" (implausibly complete) and the resulting difference, 6.77 dB, was far larger than the 2.56 dB measured for a single verified pair. | The raw-vs-processed difference was inflated; with correct matching (42 unambiguous pairs) it is 3.81 dB. |
| 6 | Linearised multilateration used for the ear-position fit. With an unknown constant time offset the bias column is nearly collinear with the position columns, so the system is ill-conditioned. | The fit returned a half-interaural distance of 11 mm (anatomically ~90 mm) and a yaw of −180°. | Replaced with nonlinear least squares (the standard GPS formulation), after which the geometry is physically sensible and the conclusion — that `hrirfilt` is already head-aligned — could be established. |

**Practice adopted as a result:** any optimisation in a new parameterisation is (a) normalised
to unit scale first, and (b) validated against a case with a known answer before its results
are believed.

---

## 7. Statement on the use of AI tools *(draft — revise before submission)*

> *Must be accurate. Revise to reflect the final state of the work.*

Generative AI tools (Anthropic Claude) were used throughout this project in the following
ways:

**Used for.**
- *Explanation and tutoring.* Deriving and explaining background theory — the
  Kirchhoff–Helmholtz boundary integral equation, Green's identities, stationary-phase
  asymptotics — and generating self-test material for my own study. The derivation document
  `notes/BIE_derivation_from_scratch.tex` was drafted with AI assistance and then checked by
  me line by line against standard references.
- *Code generation for analysis.* Diagnostic and analysis scripts in `analysis/` were
  drafted with AI assistance. All were executed by me; all reported numbers are from my own
  runs.
- *Literature triage.* Identifying and summarising related work, in particular the
  comparison against Zhang, Wang & James (ICASSP 2021). Papers cited in the thesis were read
  by me in full.
- *Editing and structuring* of written material, including this log.

**Not used for.**
- Generating results, data or figures that were not produced by executing code on real data.
- Writing thesis text presented as my own analysis without my verification.

**Verification and limitations.** AI-generated analysis code contained substantive errors,
four of which are documented in §5 of this log. In each case the error produced a plausible
but incorrect conclusion, and each was identified by consistency checks — dimensional
reasoning, rank bounds, and validation against known answers. This experience informed the
practice recorded at the end of §5. I take responsibility for all results reported in this
thesis; where AI assistance was used, the output was verified before being relied upon.

---

## Appendix — file map

```
ELEC5021/
├── PROJECT_LOG.md                          ← this file
├── ICASSP_Project_Briefing_CN.md           project briefing / plan
├── notes/
│   ├── BIE_derivation_from_scratch.tex/pdf boundary integral equation, from first principles
│   └── BIE_quiz.tex/pdf                    45 concept questions + answers
├── analysis/                               diagnostic scripts (see §4)
├── HRTF_Prediction_Neural_BEM_Approach/    method spec + work plans (supervisor)
├── SYMARE/                                 measured + BEM HRIRs, meshes, new CAR Lab archive
├── FFM/                                    trained spherical autoencoder + notebooks
└── Paper/                                  reference papers
```

---

# Appended 2026-08-28 — Unit B work, weeks of 18 and 25 August

> Everything below is additive. Nothing above this line has been edited. Where a
> statement made earlier in this log is now known to be wrong, it is listed in
> **§8, "Corrections to earlier sections of this log"**, at the end of this append —
> the original wording is left in place so the change of view is visible.

---

## 2A. Dated work log *(continuation of §2)*

### 2026-08-21 — CAR Lab bandwidth: the 16 kHz ceiling is a filter, not a measurement

**Question.** Cross-database mean-spectrum plots showed CARLab behaving completely
differently from HUTUBS and SONICOM above 16 kHz. The log previously recorded this as a
property of the measurement. I set out to confirm it from the most primitive artefact
available rather than from the delivered files.

**Method.** Two binary formats were reverse-engineered so the delivered and the raw
recordings could be compared directly (`analysis/band_limit_audit.py`):

| Format | Rate | Layout |
|---|---|---|
| `hrirfilt` | 48 kHz | `int16 L` header, then records of `int16 az, int16 el` + `2 × L float32` |
| `.imp` | **80 kHz** | records of 12 bytes + 2048 `float32`; azimuth/elevation stored as `float32` degrees with the low 16 bits zeroed |

The `hrirfilt` azimuth convention is negated relative to the export convention; this is
applied in the reader.

**Result.**

- All **259/259** `hrirfilt` archives fall off a cliff at **16 312.5 Hz**. The edge is
  identical to the bin across every file, which is the signature of a filter-design step,
  not of a transducer or an anechoic-chamber limit.
- The raw 80 kHz `.imp` recordings retain **38–40 dB SNR from 16 kHz to 24 kHz**.

**Consequence.** The usable bandwidth of the CAR Lab archive is set by the CAR Lab
filter-design stage and is recoverable. This changes what can be claimed about the
high-frequency behaviour of the model. **Action:** request the full `HRIRs/AcousticRaw/`
archive from Prof. Jin — only `hrir1` and `hrir218` are present locally.

*(This supersedes the earlier statement that "above 16 kHz the raw data rolls off to
−42 dB" — see §8, correction 7.)*

---

### 2026-08-22 — Model v2.3: three databases, FFM + FiLM only

**Scope decision.** The multi-database machinery accumulated in v2.1 (row-expanding FiLM
tables, optimiser-state rebuilds, per-group parameter groups, trunk freezing, an auxiliary
loss weight) was removed. v2.3 keeps exactly two mechanisms: **Fourier feature mapping over
frequency** in the decoder, and **per-dataset FiLM**. The training cell went from 167 to
78 lines. The reason is not aesthetic: every additional mechanism is a confound in a
comparison whose effect size turned out to be below the noise floor (see below).

**Data.** 713 subjects: CARLab 218, SONICOM 399, HUTUBS 96. Spectra are 93 frequency bins
at $\Delta f = f_s/256 = 172.27$ Hz, i.e. DC to 15.85 kHz, on HEALPix `nside=16`
(3072 directions).

**FiLM design.** Per-dataset, **per-frequency only, never per-direction** — 93 degrees of
freedom per database. Parameters are cross-dataset centred, so the transform is the
identity when a single database is present:

```python
lg = self.log_gamma - self.log_gamma.mean(0, keepdim=True)
bt = self.beta     - self.beta.mean(0, keepdim=True)
```

The encoder applies the **inverse** FiLM before encoding, tied to the decoder's parameters,
so the correction adds no new weights and the latent is dataset-neutral by construction.
$\beta$ is initialised from the measured per-database mean spectra rather than from zero.

**Results.**

| Configuration | Epochs | Subjects | Val RMSE | Baseline | Reduction |
|---|---|---|---|---|---|
| CARLab only | 150 | 218 | **1.403 dB** | 5.053 dB | 72.2% |
| 3 databases, pooled | 100 | 713 | 1.504 dB | — | — |
| — CARLab partition | 100 | 218 | **1.372 dB** | 5.053 dB | 72.8% |
| — SONICOM partition | 100 | 399 | 1.454 dB | 4.977 dB | 70.8% |
| — HUTUBS partition | 100 | 96 | 1.813 dB | 5.162 dB | 64.9% |

**What may and may not be claimed.** The validation set's own resolution was measured by
splitting it into two halves of 22 ears each and comparing: **0.19 dB**. The CARLab
single-database and pooled figures differ by 0.031 dB. **Pooling therefore cannot be
claimed to have improved reconstruction.** What *can* be claimed is that pooling three
databases with heterogeneous conventions did not degrade any partition, using 93 free
parameters per database — which is the practically useful statement.

**Sampling.** Mixing is controlled by a temperature rule, isolated in its own cell so the
mix can be changed without re-running the data pipeline:

$$P(d) \propto w_d \cdot n_d^{\tau}, \qquad \tau = 0.5$$

At $\tau=0.5$ HUTUBS is over-sampled relative to its share of subjects, which is why its
partition result is usable at all with 96 subjects.

---

### 2026-08-23 — Diagnostics: within the trained band, the bottleneck is not the autoencoder

Three diagnostics were run against the 218-subject model. **All three are confined to
DC–15.85 kHz** — 93 bins at $\Delta f = f_s/256 = 172.27$ Hz with $f_s = 44.1$ kHz. Nothing
below is a statement about the band above 16 kHz.

**Per-band residual, as a fraction of the per-band baseline.**

| Band | residual / baseline |
|---|---|
| DC and < 500 Hz | 59.3% — worst |
| 0.5–6 kHz | ~40% |
| **6–12 kHz (pinna notches)** | **25.3% — best in the table** |
| 12–16 kHz | ~35% |
| above 15.85 kHz | **not modelled — outside the trained band** |

Note the trend at the top of the band: the ratio rises from 25.3% at 6–12 kHz to ~35% at
12–16 kHz. Whatever happens above 16 kHz, it is not being extrapolated from a flat trend.

**Amortisation gap.** Replacing the encoder's output with a per-subject optimised latent
improves validation by **0.072 dB (5.1%)**. Cold-start versus warm-start optimisation differ
by 0.005 dB. Within this band the encoder is therefore not the limiting component; it
supplies an initialisation already near the decoder's optimum. (v1 measured 0.168 dB / 7% —
same conclusion, different data.)

**What this does and does not say about §1.2.**

The §1.2 premise — "Thread B's reconstruction of high-frequency pinna notches is weak
(~2–3 dB residual)" — was recorded against the **48-subject v1 model, which ran 129 bins at
$f_s = 48$ kHz, i.e. DC–24 kHz** (`analysis/amortization_gap_v2.py`: `NBINS, FS = 129, 48000`).
The present model covers DC–15.85 kHz. The two share neither a frequency axis, nor a subject
population (48 vs 218), nor a processing pipeline — raw versus processed differ by 3.81 dB on
42 matched subjects (§2, 2026-08-17 later), which also records that the v1 figures "are
**not comparable**".

**This entry therefore does not overturn §1.2.** It leaves the part of §1.2 that concerns
16–24 kHz *untested*. The notch structure that was hard to fit in v1 may still be there; it is
simply absent from the present model by construction, and the model's high-frequency behaviour
is not the same object as v1's.

What *is* wrong with the §1.2 premise is a separate, band-independent point: "2–3 dB" is an
**absolute** residual, whereas 25.3% is a **ratio to the per-band baseline**, and the two were
set against each other as if they measured the same thing. On v1's own numbers the notch band
was already the better of the two reported bands in ratio terms:

| v1, 48 subjects, DC–24 kHz | baseline | encoder residual | ratio |
|---|---:|---:|---:|
| overall | 9.697 dB | 2.326 dB | 24.0% |
| notch 6–12 kHz | 9.371 dB | 2.029 dB | **21.7%** |

The notch band looked bad in absolute dB only because its baseline is large there. So the
premise should be retired — but for *that* reason, not because a band-limited 218-subject
model disproved it.

**Consequence for the ICASSP framing.** Two rules follow.

1. **No unqualified claim about notch reconstruction, in either direction.** Neither "the
   autoencoder cannot reconstruct pinna notches" nor "the autoencoder reconstructs pinna
   notches well" may be written without the band attached. The defensible sentence is:
   *within DC–15.85 kHz, on 218 subjects, the 6–12 kHz notch band has the lowest
   residual-to-baseline ratio of any band, 25.3%.* 16–24 kHz is an open question.

2. **Thread A's argument must not rest on the notch-reconstruction claim at all.** The leg
   that is band-independent and directly measured is the downstream one: regression from
   sparse landmarks to the latent gives $R^2 \approx 0$. Thread A's value is that it computes
   notch **frequencies** from geometry exactly, $f_n \approx (2n{+}1)c/2\Delta L$, and learns
   only the smooth notch **amplitudes**. That argument holds whatever the answer above
   16 kHz turns out to be, which is precisely why it should carry the paper.

**What would settle it.** Re-run the per-band diagnostic on a model trained to 24 kHz. That
requires the full `HRIRs/AcousticRaw/` archive (open action 3): the delivered `hrirfilt` files
are digitally filtered at 16 312.5 Hz (§5A correction 7), while the raw 80 kHz recordings hold
**38–40 dB SNR from 16 to 24 kHz**. Note the symmetry of the caution — part of v1's residual
above 16 kHz may itself have been filter roll-off rather than genuine notch structure, so the
old figure is not evidence for the opposite conclusion either. Neither claim is currently
supported; the band is untested.

*(§5A correction 8 and the §8 note on §1.2 were written before this analysis and state the
overturn too strongly. They are superseded by this entry: the premise is retired, but on the
absolute-versus-ratio ground above, and only for DC–15.85 kHz.)*

---

### 2026-08-24 — Where the training time actually goes

Kernel-level timing on A100 (`analysis/FFM_2_3_bench_kernels.py`), per batch:

| Component | Time | Share |
|---|---|---|
| OSLOConv **backward** | ~1048 ms | 92% |
| OSLOConv forward | ~74 ms | — |
| DataLoader | 8 ms | 0.7% |
| **Total** | ~1136 ms | |

The backward pass costs **14.1×** the forward pass. The cause is the scatter-add over
HEALPix neighbourhoods: 658 million atomic additions per batch, which serialise. Sustained
throughput is **0.2% of the A100's BF16 peak** — this is a memory-ordering problem, not a
compute problem.

**Hardware implications.** Peak memory is 12.1 GB of 40 GB, so the 96 GB of an RTX PRO 6000
buys nothing. An H100 helps only through its 2.15× bandwidth. Fixing the kernel is worth
more than either.

**Fix.** `oslo_fast.py` replaces the scatter with a **gather** by precomputing the inverse
neighbour index ($K=9$ for `nside=16`), inside a custom `torch.autograd.Function`:

```python
gg = torch.einsum('bon,ocq->bcqn', grad_out, weight)
gg = torch.cat([gg.reshape(B, C, Q * N), gg.new_zeros(B, C, 1)], dim=2)
grad_x = gg[:, :, inv].sum(dim=2)          # gather, not scatter
```

Validated in float64 against the reference implementation: forward `max|Δ| = 0.000e+00`,
`grad_x max|Δ| = 4.44e-16`, `torch.autograd.gradcheck` passes. Expected speed-up ≈4×.
Not yet integrated into the training notebook.

---

### 2026-08-24 — Interruptible training with exact schedule continuity

Colab sessions are reclaimed, so training must survive interruption without changing the
optimisation trajectory. The training cell now takes an **absolute** target
(`TOTAL_EPOCHS = 150`, not "how many more"), writes checkpoints atomically
(`torch.save` to `.tmp` then `os.replace`), and stores optimiser state.

The subtle part is the learning-rate schedule. Reconstructing
`CosineAnnealingLR(..., last_epoch=done-1)` is **wrong** — it leaves the rate at 1e-4 where
it should be 2.94e-5, because the scheduler's anchor is taken from the current group `lr`
rather than the original one. The correct procedure is to restore `initial_lr` explicitly
and then step the fresh scheduler forward:

```python
for pg in optimizer.param_groups:
    pg['lr'] = pg['initial_lr'] = base_lr
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
for _ in range(skip):
    scheduler.step()
```

**Verification.** 150 epochs run uninterrupted versus the same run interrupted three times
(at epochs 30, 77, 120) give a **maximum learning-rate difference of 0.000e+00**.

Checkpoints are 1.2 GB: 100.9 M parameters × 4 bytes × 3 tensors (weights plus Adam's two
moment estimates). This is disk *occupancy* and is overwritten in place; the ~120 GB figure
that appeared in an earlier note was cumulative write *traffic* over a run, which is a
different quantity and was a misleading way to state it.

---

### 2026-08-25 — Duplicate audit across four databases

`HRTF_DupCheck.ipynb`. Motivation: a duplicated subject inflates the effective sample size
and can place the same person in both training and validation.

**Method.** Pairwise RMS distance and correlation between subjects over selected
directions, computed twice: on the raw spectra, and after removing each subject's own mean
level. The second pass is necessary — a synthetic test in which subject 40 was set to
subject 9 plus a uniform 8 dB gave a raw RMS of 8.000 dB (invisible against the population
spread) and a level-free RMS of 0.000.

**Threshold.** Expressing the threshold in standard deviations was abandoned: a certain
duplicate sat at only 2.1σ but at 0.7% of the median distance. The operative criterion is
the **discontinuity in the sorted distance list**, not any absolute cut — an absolute 25%
cut would have missed HUTUBS's real duplicates at 28.8% and 40.7%.

**Findings.**

| Database | Result |
|---|---|
| CARLab (218) | clean |
| SONICOM (399) | clean |
| SYMARE61 | clean |
| **HUTUBS (96)** | **2 duplicate pairs**: `pp1`/`pp96` (the FABIAN dummy head, measured twice) and `pp22`/`pp88` (the same human). Both are documented in the HUTUBS release notes; the audit recovered them independently. |

**Effective HUTUBS N = 93** — both members of the dummy-head pair excluded, one member of the
human pair. The trained model saw 96; quote 93 wherever the count is used as a statement of
statistical power.

**What the duplicates actually cost — measured, not estimated.** Both pairs land wholly on one
side of the frozen split, so there is **no train/test leakage anywhere**:

| Pair | Where | Consequence |
|---|---|---|
| `pp1` / `pp96` (FABIAN dummy head) | both in **train** | one item over-sampled among 58 training subjects; val and test untouched |
| `pp22` / `pp88` (the same human) | both in **test** | that subject carries double weight in the HUTUBS test average |

An earlier version of this entry said the **val** figure of 1.813 dB is optimistic because of
the duplicates. That is wrong: HUTUBS val is 14 subjects and contains none of `pp1`, `pp96`,
`pp22`, `pp88`, so 1.813 dB is unaffected. What is affected is the HUTUBS **test** partition,
and by this much — recomputed per ear from `_ALL_subject_errors.csv`, no retraining involved:

| HUTUBS test partition | ears | RMSE |
|---|---:|---:|
| as reported | 48 | 1.7680 dB |
| dropping `pp88` | 46 | 1.7707 dB (+0.0027) |
| dropping `pp22` | 46 | 1.7723 dB (+0.0043) |
| pooled over all three databases, dropping `pp88` | 210 | 1.4844 dB (−0.0022) |

The reported figure is therefore optimistic by **at most 0.004 dB**, against a measured
validation resolution of **0.19 dB** — about **2% of the noise floor**, and roughly 30× below
the 0.13 dB scale at which val and test agree. It is not a reportable quantity.

**Action (revised — no retraining).**

1. **Do not rebuild the split.** Deleting the HUTUBS entry from `split_v21.json` re-randomises
   the HUTUBS partitions and voids the frozen test set — which is exactly what makes
   "single-database vs three-database on the same CARLab test list" a clean comparison.
   Spending that to correct 0.004 dB is a bad trade.

2. **Correct by recomputation, not by retraining.** Deduplication at evaluation time is exact
   and costs nothing; the per-ear CSV already carries every ear. Report the HUTUBS test figure
   as **1.768 dB (1.771 dB deduplicated, 46 ears)** and label which is which. The same applies
   to any future table: drop the row, do not retrain the model.

3. **Keep both pairs where they are — they are an asset, not a defect.** `pp22`/`pp88` is one
   human measured twice and `pp1`/`pp96` is one manikin measured twice. Encoding both members
   and measuring the latent distance, normalised by the median population pairwise distance,
   gives the **measurement-repeatability ceiling** for the geometry→latent stage: no geometry
   predictor can beat it, and without it there is no way to say whether $R^2 = 0.3$ is poor or
   near the ceiling. First indication — the two members of the human pair already reconstruct
   almost identically: `pp22` 1.7134 / 1.6189 dB versus `pp88` 1.7157 / 1.6949 dB (left/right),
   the left ears agreeing to 0.002 dB.

4. **Keep the cross-split leak check in the reporting cell**, where it runs on every test
   evaluation and prints the side of each member. It must keep failing loudly: an earlier
   version matched a bare `pp1` against IDs carrying the `HU_` prefix, found nothing on either
   side, and treated "not found" as "same side, no leakage" — printing a pass. Match by suffix
   with a word boundary, so that `pp1` cannot match `pp19`.

---

### 2026-08-26 — Mapping the 61 SYMARE subjects into the CAR Lab archive

**Why this matters.** 3-D meshes exist only for the 61 original SYMARE subjects. Whichever
CAR Lab subjects correspond to them is exactly the training set available to
Thread B's downstream geometry→latent stage. This is the number that caps that experiment.

**Two independent evidence chains.**

1. **Names.** `carlabdb` stores real names; SYMARE files are `HRIR_<surname>.mat`, and
   `SymareHrirs/HRIR_<name>.mat` is byte-identical to `HRIRs/Acoustic/HA<nn>.mat`. This
   yields 51 unambiguous surname pairs.
2. **Spectra.** Correlation of HEALPix-interpolated magnitude between the two pipelines.

Using the 51 name pairs as ground truth, the spectral method's rank-1 accuracy was measured
under 2×2 hypotheses (sampling rate × median-plane mirror). Chance is 1/218 = 0.46%.

| SYMARE61 $f_s$ | Mirrored | rank-1 |
|---|---|---|
| 48 000 | no | 68.6% |
| 48 000 | yes | 27.5% |
| **44 100** | **no** | **92.2%** |
| 44 100 | yes | 82.4% |

The SYMARE61 `.npy` files had already been resampled to 44.1 kHz for multi-database
training; the comparison had been run at 48 kHz, stretching the frequency axis by 8.8% and
displacing a 12 kHz notch by about 1 kHz. This is why visual inspection had suggested that
"almost nothing lines up". *(See §8, correction 9.)*

**Calibrated threshold.** With the corrected axis, $r \ge 0.45$ gives 47/47 correct and
$r < 0.45$ gives 0/4 — a clean separation with no transition band. An earlier
`R_HIGH = 0.70` was needlessly strict.

**Uniqueness check.** Enforcing injectivity caught `CL178` as the top-1 for two different
SYMARE subjects: `HA19` at $r = 0.637$ with a margin of 0.270, and `HA04` at $r = 0.295$
with a margin of 0.007. The second is noise; `HA04` was assigned from its name instead.

**Outcome — `analysis/symare61_to_carlab_RESOLVED.csv`.**

| Tier | Criterion | Count |
|---|---|---|
| A1 | name and spectrum agree | 47 |
| A2 | name confirmed against `CARLab_subjects.txt` after inspection | 5 |
| B | no name, spectrum strong and well separated | 2 |
| C | name and spectrum conflict, spectrum strong | 3 |
| D | unresolved | 4 |

**57 of 61 resolved, 52 of them high-confidence.** Unresolved: `HA18`, `HA29`,
`HA47`, `HA48`. `HA18` is the interesting one — the name candidate
`CL003` is ranked 88th of 218 by spectrum, where 47 of 51 verified pairs rank 1st, so the
name probably belongs to a different person.

**The operative number for Thread B: 57 CAR Lab subjects have a 3-D mesh available.**

---

### 2026-08-28 — Per-subject error reporting

Requested by the supervisor. The distinction matters and is recorded here because the two
numbers are easy to confuse.

The value printed each epoch during training pools **every** element of the validation
split into one RMSE. A per-subject error computes the same quantity within one subject.
They are related by

$$\mathrm{RMSE}_{\text{pooled}} \;=\; \sqrt{\;\overline{\mathrm{RMSE}_s^{\,2}}\;}$$

— the **quadratic** mean over subjects, not the arithmetic mean. By Jensen's inequality
$\overline{\mathrm{RMSE}_s} \le \mathrm{RMSE}_{\text{pooled}}$, and the size of the gap is a
measure of how unevenly the error is distributed across people. Reporting the arithmetic
mean of per-subject errors alongside the pooled figure and calling them the same thing
would be an error; both are reported, labelled.

A third convention appears in the literature: **log-spectral distortion** takes the RMS over
frequency first and then the *arithmetic* mean over directions, so it is systematically
lower again. Comparisons with published numbers must use this one.

`FFM_2_3_cell9_bandmaps_zip.py` writes, per subject: 19 HEALPix triptychs (ground truth,
prediction and $|$error$|$ side by side, one per 5-bin band, 172.27 Hz × 5 = 861 Hz wide),
an error-versus-frequency curve, and a per-band table; plus one CSV over all subjects
carrying `rmse_band`, `rmse_all`, `lsd`, `mae`, `bias`, `rmse_notch` and `peak`.

---

### 2026-08-29 — Repository published; data provenance audited

**Repository.** `aply777/hrtf-capstone`, private: 1953 files, 359.5 MB, one commit. Code,
analysis scripts, the training notebook (translated to English, outputs and figures kept),
`docs/PROJECT_LOG.md`, and two demonstration volumes of the per-subject error maps.
Weights are a Release asset (`v2.3`, `FFM23_best.pth`, 405 MB) rather than a commit: both
checkpoints exceed GitHub's 100 MB per-file limit, and Git LFS's free tier (1 GB storage,
1 GB/month bandwidth) is smaller than `FFM23_latest.pth` alone.

**Baselines — two separate facts.** The per-database validation baselines were already
being printed by Cell 3 of the training notebook and had simply never been transcribed
into any table:

| | val baseline | model | reduction |
|---|---:|---:|---:|
| CARLab | 5.053 dB | 1.372 | 72.8% |
| SONICOM | **4.977 dB** | 1.454 | 70.8% |
| HUTUBS | **5.162 dB** | 1.813 | 64.9% |

The **test** baselines, however, have never been computed: Cell 9 computes `te_base` only
under `RUN_TEST = True`, and the locked metrics come from Cell 10, which reports no
baseline. `notebooks/Compute_Baselines.ipynb` now computes both, without loading the model
or touching the GPU, and re-derives the three validation figures above as a check that the
convention is reproduced exactly.

The convention is worth restating because it was queried and misremembered. The baseline
is built from the **training** mean only — a mean that has seen the val or test subjects
is not a prior — and it is a **pooled** RMSE over (ears x bins in band x directions),
which is the same reduction `evaluate()` performs. The arithmetic mean of per-ear RMSEs is
a different and systematically smaller quantity (see §5A correction 12 territory: the
test-set numbers differ by 1.4716 vs 1.4867), and substituting it would silently inflate
every "reduction" percentage. The new notebook prints all three variants side by side so
the distinction cannot be lost again.

**CARLab baseline verified locally, and the first test baseline.** The three validation
baselines above came from the stored Colab output and had never been re-derived. All the
inputs are on the development machine — the 44.1 kHz CARLab export, `healpix_mirror_idx.npy`,
and `split_v21.json` (which is committed as `analysis/split_v21.json`; an earlier note in
this entry claiming it lives only on Drive was wrong) — so CARLab was recomputed from
scratch with the training convention: training-set mean, right ears mirrored, 93 bins,
pooled RMSE.

| CARLab partition | ears | pooled | arithmetic | std across ears |
|---|---:|---:|---:|---:|
| val | 44 | **5.0531** | 4.9323 | 1.0980 |
| test | 44 | **4.7724** | 4.7386 | 0.5678 |
| train, in-sample | 348 | 5.0820 | — | — |

Val reproduces the recorded 5.053 to **+0.0001 dB**, which confirms the convention.
Test 4.7724 is the project's first test-set baseline: against it the model's 1.2909 dB is
a **73.0%** reduction.

Two things this settles.

*The test baseline is not the validation baseline.* They differ by 0.28 dB — the 22 test
subjects are simply less variable than the 22 validation subjects (std across ears 0.57
against 1.10). Quoting the validation baseline against a test model figure would give
74.5% instead of 73.0%. The rule that a baseline is scored on the same partition as the
model number beside it is therefore not pedantry; it is worth a point and a half here.

*The in-sample correction is invisible at this sample size, and an earlier claim in this
project's notes was wrong on the sign.* For a mean estimator the expected in-sample and
out-of-sample MSEs are `sigma^2 (1 -/+ 1/n)`, predicting roughly 0.03 dB, with in-sample
the lower. Measured, the in-sample figure is 5.0820 — **higher** than validation's 5.0531.
The 1/n bias is swamped by between-partition sampling noise, which this project already
measured at 0.19 dB on a 44-ear set. The reason to score out-of-sample is that the
comparator must match the model's partition, not that the bias is large; it is not.

**Provenance audit.** Every database reaches its `(129, 3072)` array through the same
three steps (FFT, `20 log10`, spherical thin-plate spline onto HEALPix `nside=16`), but
the readers in front of that projection were scattered. Two problems, one fixed outright
and one that turned out not to be a problem:

- `spherical_spline.py` — the projection every export path calls — **was not committed**,
  and the four scripts importing it did so through a hardcoded, stale absolute path
  (`/sessions/exciting-jolly-cannon/...`). Those four scripts could not have run on any
  machine. Fixed: the module is now `analysis/spherical_spline.py`, and the import path
  resolves relative to `__file__`.
- The SOFA reader appeared to be missing — nothing in `ELEC5021/` mentions SOFA at all. It
  exists, in `26S1/ELEC5020/`, and is now committed as
  `analysis/preprocess_multidataset_stps.py`. It covers SYMARE (`.mat`, 48 kHz, resampled),
  HUTUBS and SONICOM (SOFA), leaving CARLab on its own `hrirfilt`/`.imp` path. All four
  databases are therefore reproducible from this repository.

Four properties of that reader are worth recording, none of them invalidating the archives
in use: one STPS matrix is built per database from the **first subject's** direction grid
and reused, which is safe only for single-grid databases (CARLab is not one, and caches
per grid); its dB floor is `1e-6` against the CARLab path's `1e-9`, a difference that does
not appear to bind since the lowest magnitude measured in CARLab is about −80 dB; SONICOM
is read from `FreeFieldCompMinPhase` while HUTUBS is read raw, which is a genuine
processing-convention difference between the two and part of what FiLM absorbs; and
impulse responses longer than 256 samples are truncated rather than windowed.

**Path portability.** Seven scripts carried an absolute `ROOT` baked in
(`/sessions/exciting-jolly-cannon/mnt/ELEC5021/...`), a container mount path that exists
on no machine, including the one this project is developed on. Two of them also wrote
their figures and `.npz` outputs to that path. All now resolve the data root through
`analysis/_paths.py` — `$ELEC5021_ROOT`, else the parent of the repository, else
`~/Documents/26S2/ELEC5021` — and fail at startup naming every path tried, rather than
several minutes into a batch job. Outputs are written beside the script.

**CARLab chain documented.** The four steps and their order were nowhere written down:
`audit_carlab.py` (267 records -> 259 files -> 241 readable -> **218** distinct people),
`export_carlab_npy.py` (hrirfilt -> HEALPix at 48 kHz), `finalize_carlab.py`
(CL001-CL218, mirror index), and `export_carlab_44k1.py`, which is the one training
actually reads. Step 4 re-exports from the binary rather than resampling step 2's output,
because resampling belongs in the time domain. `carlab_export.py`, the early draft of
step 2, is moved to `analysis/archive/` with a header saying so.

**Also added.** `docs/DATA.md` — the four Drive archives, the expected Drive layout, the
44.1 kHz rationale, and a standing note that the CARLab archive is unpublished lab data
whose link must be removed from both the file and the git history before this repository
could ever be made public.

---

## 4A. Experiment index *(additions to §4)*

| Script | Question | Headline result |
|---|---|---|
| `analysis/band_limit_audit.py` | Is CARLab's 16 kHz ceiling a measurement limit or a filter? | **A filter.** 259/259 archives cliff at 16 312.5 Hz; raw 80 kHz recordings hold 38–40 dB SNR from 16–24 kHz |
| `FFM_2_3_MultiDB.ipynb` | What does three-database joint training achieve? | 713 subjects, pooled 1.504 dB; CARLab partition 1.372 dB (1.403 single-database — difference within noise) |
| `analysis/FFM_2_1_diagnostics.py` | Which band carries the error? | Notch 6–12 kHz residual/baseline **25.3%**, best in the table; DC and <500 Hz worst at 59.3% |
| `analysis/FFM_2_1_amort_fast.py` | Encoder or decoder? | Gap **0.072 dB (5.1%)**; cold−warm 0.005 dB — the encoder only supplies an initialisation |
| `analysis/FFM_2_1_latent_structure.py` | How many numbers must the downstream stage predict? | *(not yet run — blocks the geometry→latent work)* |
| `analysis/FFM_2_3_bench_kernels.py` | Where does training time go? | OSLOConv backward = **14.1×** forward, 92% of the epoch; 0.2% of A100 BF16 peak |
| `oslo_fast.py` | Can the atomic scatter be eliminated? | Inverse index turns scatter into gather; `gradcheck` passes, gradient agrees to 4.4e-16 |
| `HRTF_DupCheck.ipynb` | Duplicates in any of the four databases? SYMARE↔CARLab correspondence? | HUTUBS has exactly **2 pairs** (matching the published release notes); cross-database rank-1 **92.2%**, 57/61 resolved |
| `FFM_2_3_cell9_bandmaps_zip.py` | Per-subject, per-band spatial error | Per-subject error tables and 19 band triptychs per subject |

---

## 5A. Corrections register *(additions to §5 — numbering continues)*

| # | Error | How caught | Effect on conclusions |
|---|---|---|---|
| 7 | Recorded that "above 16 kHz the raw data rolls off to −42 dB". What rolls off is the delivered `hrirfilt`; the **raw 80 kHz recordings hold 38–40 dB SNR from 16–24 kHz**. | Cross-database mean spectra showed CARLab behaving unlike the other two above 16 kHz, prompting a reverse-engineering of the `.imp` format and direct measurement of the raw recordings. | "16 kHz is the measurement ceiling" is withdrawn; it is a filter-design artefact. The judgement about usable bandwidth changes, and a new action follows (obtain the full `AcousticRaw` archive). |
| 8 | Recorded that Thread B reconstructs pinna notches poorly (~2–3 dB residual), and used this as the motivation for Thread A. | Per-band diagnostics on the 218-subject model put the notch band at 25.3% residual/baseline — the **best** band in the table. The earlier impression came from a 48-subject model. | The motivation for Thread A is rewritten: not "the autoencoder cannot reconstruct notches" but "the autoencoder reconstructs them, and geometry cannot predict them". This is the single most consequential wording change for the paper. |
| 9 | The SYMARE61 sampling rate was set to 48 kHz in the cross-database comparison, but those `.npy` files had already been resampled to 44.1 kHz for multi-database training. The frequency axis was stretched by 8.8%, displacing a 12 kHz notch by ~1 kHz. | Visual inspection suggested almost no pair matched. Rather than trusting the eye, the 51 **independent** surname pairs were used as ground truth and rank-1 accuracy was computed under all four (rate × mirror) hypotheses. | Rank-1 rose from 68.6% to **92.2%**; `HA02` and `HA07`, previously wrong at top-1, corrected themselves. The suspicion that CARLab and SYMARE shared no subjects was eliminated. |
| 10 | The calibration cell computed similarity matrices under all four hypotheses and selected the best, **but never exported the selected matrix**; downstream cells silently used a left-over mirrored version. | The final table's name-agreement rate was 82.4% — **bit-for-bit identical** to the `44100 / mirrored` row of the calibration table. A coincidence that should not occur. | Five correct correspondences were missing from the final table. **Lesson: having selected the right answer, it must be propagated explicitly, or the selection did nothing.** |
| 11 | Surname matching was treated as reliable ground truth. It fails in three distinct ways: initials-only records, spelling variants of one surname, and several people sharing a surname (`CL108` and `CL109` were the top two candidates for the same subject). | The uniqueness check flagged `CL178` as top-1 for two different SYMARE subjects with margins of 0.270 and 0.007. | Two independent evidence chains should be used as **mutual leads**, not intersected. Intersecting them discards correct answers that only one chain can see; using each to generate candidates for the other recovered 5 further pairs and rejected 1 spurious one. |

**Additional practice adopted.** Where an **independent evidence chain** exists (names
versus spectra, here), it is used to **calibrate the pipeline's parameters**, not merely to
sanity-check the output by eye. Visual agreement is not a valid criterion in the presence of
a known systematic difference: the two pipelines here differ by 3.52 dB in a
direction-dependent way, so even a correct pair will not overlay.

---

## 8. Corrections to earlier sections of this log

The sections above this append were left unedited. The following statements in them are
superseded.

**§1.1, Thread B row — "Trained; under diagnostic evaluation".**
Now: v2.3 trains three databases jointly (713 subjects) with per-dataset FiLM.
Reconstruction has converged — CARLab validation 1.372 dB against a 5.053 dB baseline — and
the diagnostics are complete: the bottleneck is no longer the autoencoder. The open item is
the downstream geometry→latent stage, which has not been started.

**§1.2, "Why Thread A" — "Thread B's reconstruction of high-frequency pinna notches is weak
(~2–3 dB residual)".**
This premise is **wrong** and must not be reused in the paper. The 6–12 kHz notch band has
the *lowest* residual/baseline ratio of any band at 25.3%; the 2–3 dB figure came from a
48-subject model. The second half of §1.2 — that the regression from sparse landmarks gives
$R^2 \approx 0$, and that Thread A computes notch frequencies from geometry and learns only
their amplitudes — remains correct and is now the *whole* of the argument. See §5A,
correction 8.

**§1.3, Deadlines.** Still 16 September 2026. Added: the three-database results usable in
the paper now exist, and the autoencoder-side diagnostics are complete, so tuning on that
side can stop.

**§2, entry of 2026-08-17 (later), "Provenance of the existing HA01–61" and the
raw-versus-processed discussion.** The bandwidth statement in that entry is superseded by
§5A correction 7; the 3.81 dB raw-versus-processed figure and the head-alignment conclusion
are unaffected.

**§5, count of errors.** Now eleven, not six.

**§7, Statement on the use of AI tools — "AI-generated analysis code contained substantive
errors, four of which are documented in §5".**
The count is now eleven, and the errors are worth separating by origin:

> Of the eleven errors recorded in §5 and §5A, some originated in AI-generated analysis code
> (an optimiser learning rate mismatched to the parameter scale; a string-handling fault that
> mis-matched every subject; a selected result that was computed but never propagated), and
> others originated in my own incorrect assumptions about the data (reading a filter-imposed
> bandwidth limit as a measurement limit; supplying the wrong sampling rate; treating surname
> matching as reliable ground truth). What the two classes have in common is that each
> produced a plausible but wrong conclusion, and that each was found by a consistency check —
> dimensional reasoning, rank bounds, comparison against a known answer, cross-validation
> against an independent evidence chain — rather than by intuition.

**§Appendix, file map.** Add `analysis/symare61_to_carlab_RESOLVED.csv` (the 61→CARLab
correspondence), `oslo_fast.py`, and `HRTF_DupCheck.ipynb`.

---

## 9. Open actions carried forward

| # | Action | Blocks |
|---|---|---|
| 1 | Run `analysis/FFM_2_1_latent_structure.py` | Fixes the target dimension $K$ for geometry→latent; **blocks all downstream work** |
| 2 | Request the full `HRIRs/AcousticRaw/` archive from Prof. Jin | Any claim above 16 kHz |
| 3 | Report the HUTUBS test figure deduplicated (1.771 dB, 46 ears) alongside 1.768 dB. **No retraining** — the duplicates are same-side, worth ≤0.004 dB; keep them as the repeatability ceiling (2026-08-25) | The HUTUBS row of any results table |
| 4 | Assert that HUTUBS and SONICOM really do use one direction grid per database, rather than assuming it, in `preprocess_multidataset_stps.py` (2026-08-29) | Whether the shared STPS matrix is valid for every subject |
| 4 | Integrate `oslo_fast.py` into the training notebook | ~4× training throughput |
| 5 | Confirm that 3-D meshes physically exist for the 57 resolved subjects | The size of the Thread B training set |
| 6 | Delete `analysis/symare61_to_carlab_final.csv` (computed on the 48 kHz axis), `analysis/_to_delete/`, `FFM_2_1_multidataset_patch.py` | Housekeeping |
| 7 | **Publish the code to GitHub for the supervisor to review.** Before pushing, strip subject-identifying material — `analysis/ha_to_carlab_map.csv`, `analysis/symare61_to_carlab_RESOLVED.csv` and any `CARLab_subjects.txt` contain real participant surnames. Publish the ID mapping only as opaque IDs, or keep the repo private. | Supervisor review; research-ethics compliance |

---

# Appended 2026-09-03 — Unit B work, geometry → latent

## 2B. Dated work log *(continuation of §2A)*

### 2026-08-31 — Prior art read, and the landmark data reverse-engineered

Five regression papers were read in full (`Paper/Regression/`). The structural finding is
that methods driven **only by geometry or anthropometry** sit at **3.8–5.0 dB LSD**
(Zhao et al. 2024, mesh, 3.78; Niu et al. 2025, 23 anthropometric parameters, 4.90–5.07),
while every method reaching **2.7–3.6 dB** — NIIRF, RANF, GraphNF-SCA — uses a handful of
the *target subject's own measured HRTFs* for retrieval or LoRA adaptation, not geometry.
The field has largely moved from geometry to few-shot measurement. Zhou et al. (ICASSP
2021) is the exception at 1.38 dB SDE, but its ground truth is FDTD-simulated **from the
same mesh**, so mesh → HRTF is a deterministic function there: that number is an upper
bound on predictability, not a prediction result.

Two omissions in Niu et al. (2025) are worth recording because this project's own results
turn on them. The paper contains **no population-mean baseline anywhere** — its only
comparison is against other DNNs — and it shows the autoencoder's reconstruction
qualitatively (Fig. 3d) without ever reporting its LSD. Its four methods land within a
0.2 dB band around ~5 dB on test sets of 5 and 6 subjects, differences far below the
standard error. The paper is sound as an architecture paper and silent on whether
anthropometry carries usable information.

`SYMARE/3DShapeData.mat` was supplied without documentation of any kind; the SYMARE v0.9
readme describes the database in detail and **does not mention the file**. Its contents
were recovered from the coordinates: 61 subjects × **33 landmarks** × 3, in millimetres,
already in a common frame (centroid spread across subjects 1.5 mm, so no Procrustes step is
needed). Blocks: indices 0–12 one pinna at y ≈ +79 mm; 13–17 midsagittal (nose tip at
x = +119, vertex at z = +143, occiput at x = −106); 18–30 the other pinna in one-to-one
correspondence i ↔ i+18; 31–32 a symmetric pair. Centroid size is 510.8 ± 21.3 mm over the
61 subjects, i.e. real head-size variation that must **not** be normalised away.

**Laterality settled acoustically.** Which block is the left ear cannot be read off the
geometry. It was determined from SYMARE's own HRIRs: taking broadband-energy-weighted mean
directions of `hL` and `hR` over the 393 measured directions in Matlab `sph2cart`
coordinates gives ⟨y⟩ = **+0.495 ± 0.031** for `hL` and **−0.494 ± 0.034** for `hR`; the ILD
over the y > 0 hemisphere is **+12.34 ± 2.03 dB**, positive in 18/18 subjects; and
`dL·ê_P > dL·ê_N` in **18/18**. Block 0–12 is the left ear. That the inner products land
near ±0.85 rather than near 0 also confirms the landmark and acoustic frames are the same
frame, which the test needed as a premise. `FFM_2_4` cell 2b re-runs this and asserts.

Also read out of the SYMARE readme: the acoustic HRIRs are `hL`/`hR` at **256 × 393**,
`fs = 48000 Hz`, source at 1 m, elevation truncated at −45° — the same 393-direction grid
CARLab uses, which explains the 92.2% cross-database rank-1. The BEM sets are **two**, and
only one is broadband: `Bem/HE16kHz` (head+ears, valid to ~16 kHz) and `Bem/HTE05kHz`
(head+torso+ears, valid only to **~5 kHz**). Which of the two produced the recorded
"BEM vs measured 6.28 dB, R² = 0.54" is not stated in this log and needs checking. Finally,
the readme states that Burton–Miller simulations at 20 kHz were in progress for a
**version 1.0** — dated 2014-03-20, so the open question is whether v1.0 ever shipped.

### 2026-09-01 to 09-03 — Geometry → latent, run to its stopping rule

The pipeline the supervisor's group proposed — landmarks → random Fourier features →
regression → frozen decoder → HRTF — was implemented in `landmark2z/FFM_2_4_Landmark2Z.ipynb`
and run in the pre-registered order. Model classes are not copied into it: it `exec`s the
cell of `FFM_2_3_MultiDB.ipynb` that defines `FFM_OSLO_Autoencoder`, so the two cannot
drift.

**The side-variance figure changed by a factor of fifty.** The v1 diagnostics put 71.7% of
latent variance on the "which ear" direction. On the v2.3 checkpoint it is **1.4%** — the
mirroring in `MonauralHRTFDataset` works as the acceptance test intended. The latent is
therefore not projected onto a person subspace, and the v1-derived ceiling ("at most the
18.6% person component is predictable") no longer applies.

**Subject count and split.** 52 high-confidence mapped subjects (tiers A1+A2), which the
frozen `split_v21.json` partitions 41 / 3 / 8 into CARLab train / val / test. The partition
was not rebuilt. Every figure is reported twice: `CV-all` (10-fold over 52) and
`AE-heldout` (fit on the 41, evaluated on the 11 whose HRTFs the autoencoder never saw).

**Is there signal?** Unbiased distance correlation with a 999-draw permutation test, at
subject level so the samples are independent: **dCor²ᵤ = 0.0710 against a null 95th
percentile of 0.0451, p = 0.0060.** Negative controls: shuffled latent 0.0360 (p = 0.080),
random features 0.0236 (p = 0.202). The ear-level figure (0.1940) is inflated by
within-subject duplication and is not quoted.

**Where the signal is — the substantive result.**

| band | bins | dCor²ᵤ | p |
|---|---:|---:|---:|
| DC–0.5 kHz | 3 | −0.0400 | 0.945 |
| 0.5–6 kHz | 32 | +0.0254 | 0.180 |
| **6–12 kHz** | 35 | **+0.0862** | **0.005** |
| **12–15.8 kHz** | 23 | **+0.0980** | **0.005** |

Both survive Bonferroni over four tests. Nothing is detectable below 6 kHz. This matters
because head size — the quantity 33 landmarks most obviously encode — controls the *low*
frequency response through head shadowing. The signal is instead confined to the band the
pinna controls, and 6–12 kHz is the band the autoencoder reconstructs *best* (residual /
baseline 25.3%). The Thread A motivation now has a measured form: **the autoencoder
reconstructs the pinna notches; sparse geometry does not predict them.**

Two cautions belong with that table. The DC–0.5 kHz null rests on **3 bins**, and those
bins carry known pathology, so it is a statement about data quality rather than about
geometry. And "not detectable below 6 kHz" is not "absent" — power is low throughout at
N = 52.

**How much is usable.** Random Fourier features (m = 512) into ridge, σ and α selected by
inner cross-validation on training folds only: **CV-R² = 0.0518 ± 0.0151**, `AE-heldout`
R² = 0.0889, shuffled-target control 0.0005 ± 0.0017. The pre-registered stopping rule
(CV-R² < 0.10) fired. σ and α both landed at the low end of their grids but interior to
them, and small σ drives random features toward constants — the selected model is
effectively **linear**, so non-linearity bought nothing.

The `AE-heldout` figure being *higher* than `CV-all` is the opposite of what in-sample
latent contamination would produce, so that concern — raised when the 41/3/8 split was
first examined — is not supported.

**End to end.** Predicted latents pushed through the frozen decoder, evaluated once on the
11 held-out subjects against the mean of the 348 CARLab training ears:

| | LSD [dB] |
|---|---:|
| landmark → ẑ → HRTF | **4.208** |
| population mean | **4.253** |

The 0.045 dB gap is far below the 0.19 dB validation resolution measured on a 44-ear set,
and this set is 22 ears. **The prediction is indistinguishable from the population mean.**
"1.1% reduction" must not be quoted — nothing is resolved at that scale. Two conservative
choices are baked in and both work against the model: ẑ is predicted per subject, so one ẑ
is scored against both ears, and the real ~1.5 mm left/right asymmetry is charged to the
model as error.

This is the same test whose absence from Niu et al. (2025) was noted above. It was run
here, and geometry does not beat the baseline.

**The MLP, as a control.** `landmark2z/FFM_2_5_MLP_train.ipynb`, 2 layers × 128, dropout
0.2, AdamW, checkpointing and resume in the style of `FFM_2_3`. Trained on 33 subjects,
selected on 8, evaluated once on the same 11.

| epochs | train R² | best val R² | test R² |
|---:|---:|---:|---:|
| 500 | 0.757 | +0.0674 @ ep 39 | — |
| 5000 | **0.9916** | **+0.0674 @ ep 39** | **−0.1106** |

Ten times the training left the best validation score bit-identical, at the same epoch.
The network reproduces its 33 training subjects almost perfectly and is *worse than the
population mean* on unseen people. Capacity and training time are not what is missing.

**Meshes: none are held.** A `find` over the entire project returns **zero** `.ply`, `.obj`
or `.stl` files — not for SYMARE, not for HUTUBS, not for SONICOM. §9 action 5 ("confirm
that 3-D meshes physically exist for the 57 resolved subjects") is therefore answered in
the negative for the local archive: only `HRIRs/` was ever supplied for SYMARE, and the
HUTUBS and SONICOM meshes have not been downloaded.

**Why meshes and not more landmarks.** Recorded in `mesh2z/README.md`. Landmark features
cannot be pooled across databases the way HRTFs were: direction is a continuous coordinate,
so any measurement grid resamples onto HEALPix, whereas landmark identity is categorical
and HUTUBS's 23 named anthropometric parameters are not samples of the same field as
SYMARE's 33 coordinates. **Surface meshes restore poolability** — a mesh is a sampling of a
continuous surface, so meshes from different labs can be resampled onto a common
representation exactly as the HRTF grids were. That is what lets the subject count reach
~319 (SONICOM ~200 + HUTUBS 58 + SYMARE 61), all of them already inside the autoencoder,
which takes the null 95th percentile of dCor²ᵤ from 0.045 to about 0.012.

The defensible conclusion is therefore **not** "we need meshes" but: sparse landmarks at
this scale are insufficient, and the failure is one of sample size as much as of geometric
detail — a model fitting its training set to R² = 0.99 and generalising worse than the mean
is the signature of the former. Both must change together, and meshes are the only
representation that permits the second. The published record agrees: the mesh method that
worked used **645 subjects** (Zhou et al. 2021), while the mesh method run on **30**
subjects found detail beyond ~2048 points made results *worse* (Zhao et al. 2024) and voxel
resolution between 16³ and 64³ barely mattered.

### 2026-09-03 — Repository housekeeping

`landmark2z/` and `mesh2z/` added as self-contained sub-projects. `README.md` in each is
deliberately written without statistical vocabulary; the full conventions and caveats live
in `notes/landmark2z/RESULTS_landmark2z.md`, **outside the repository**, along with a
Chinese defence cheat-sheet.

`symare61_to_carlab_RESOLVED.csv` gained a `results/README_symare61_to_carlab.md`
documenting its schema, the five confidence tiers, the calibration that produced them, and
the fact that the SYMARE-side and CARLab-side spectra for the same person differ by
**3.52 dB in a direction-dependent way** and may never be mixed. It also records that
`results/ha_to_carlab_map.csv` is superseded — 50 resolved against 57, with **zero
contradictions**, the seven new resolutions all being rows the old file left empty.

The HEALPix mirror index had accumulated **three** locations across the project
(`MyDrive/` root, `MyDrive/HRTF_Checkpoints/` where `healpix_mirror.py` writes it, and
nowhere at all). All five call sites now search the three candidates in order and fall back
to regenerating it; the stored file was verified bit-for-bit identical to the two-line
regeneration, and both satisfy the bijection and involution assertions.

---

## 4B. Experiment index *(additions to §4A)*

| Script | Question | Headline result |
|---|---|---|
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 2b | Which landmark block is the left ear? | **Block 0–12.** Determined acoustically, 18/18 subjects; the SYMARE readme documents nothing about this file |
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 5 | Does the v2.3 latent still encode which ear? | **1.4%** of variance, against 71.7% on v1 — the mirroring works |
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 6 | Is there any landmark→latent signal? | **dCor²ᵤ = 0.0710, p = 0.0060** at N = 52; controls at 0.0360 and 0.0236 |
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 7 | Which band carries it? | **6–12 kHz (0.0862) and 12–15.8 kHz (0.0980), both p = 0.005; nothing below 6 kHz** |
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 8 | How much is usable? | **CV-R² = 0.0518 ± 0.0151**; stopping rule fired; selected model effectively linear |
| `landmark2z/FFM_2_4_Landmark2Z.ipynb` cell 10 | Does it beat the population mean end to end? | **No.** 4.208 dB against 4.253 dB — a 0.045 dB gap against 0.19 dB resolution |
| `landmark2z/FFM_2_5_MLP_train.ipynb` | Does more capacity rescue it? | **No.** train R² 0.9916, test R² **−0.1106**; 500 and 5000 epochs give bit-identical best validation |

---

## 5B. Corrections register *(additions to §5A — numbering continues)*

| # | Error | How caught | Effect on conclusions |
|---|---|---|---|
| 12 | Guidance and code assumed the v1 figure that **71.7%** of latent variance encodes "which ear", and `geo_latent_fit.py` printed a warning telling the reader that anything far from 71.7% meant the side labels were wrong. | Measured on the v2.3 checkpoint: **1.4%**. Under the old logic the correct answer would have been diagnosed as a labelling bug. | The v1 ceiling ("at most the 18.6% person component is predictable") is withdrawn. `person_subspace()` now decides from the measurement — projecting only above 5% — and warns above 50% instead. |
| 13 | The geometry notebook called `model.encoder(x, dataset_id)`. | `TypeError: forward() takes 2 positional arguments but 3 were given`. | Had the signature happened to match, this would have been a **silent** error: the encoder takes only `x`, and the per-dataset correction is an **inverse FiLM applied before it** inside `FFM_OSLO_Autoencoder.forward`. Calling the encoder directly compiles the measurement colouring into `z` — precisely what `z` is meant to be free of, and what the downstream stage consumes. The notebook now reproduces the correct path and asserts it is bit-identical to `model.forward`. |
| 14 | The test for whether AE-train latents differ from AE-heldout latents was run over **ears** (82 vs 22), reporting t = +2.14, p = 0.035 and concluding the `AE-heldout` figure should be preferred. | Left and right ears of one subject correlate at 0.978 after mirroring, so the effective N is 41 and 11, not 82 and 22. | The test overstates N twofold and the significance does not survive at subject level. It is now computed per subject, with the ear-level figure printed alongside and labelled not-for-quotation. The concern was independently settled by the `AE-heldout` R² coming out *higher* than `CV-all`. |
| 15 | `FFM_2_5` printed "MLP beat ridge (0.0674 vs 0.0518)". | The 0.0674 is the **maximum over 5000 epochs** of a score computed on **8** subjects; 0.0518 is a 10-fold cross-validated mean over 52. Maximising a noisy statistic over 5000 draws produces a positive number from nothing. | Not a result. The comparable figure is the MLP's test score, which involved no selection: **−0.1106**. Ridge's +0.0518 stands as the best result on this problem. |

---

## 8B. Corrections to earlier sections *(continuation of §8)*

**§9 action 5 — "Confirm that 3-D meshes physically exist for the 57 resolved subjects".**
Answered, negatively, for the local archive. **No mesh file of any format exists anywhere in
the project.** Only SYMARE's `HRIRs/` was supplied; the `Meshes/` tree documented in the
v0.9 readme was not. HUTUBS's and SONICOM's meshes are publicly downloadable and have not
been fetched. Everything in `landmark2z/` is therefore a bound on what **33 landmark
points** can do at **N = 52**, and says nothing about dense geometry.

**§9 action 1 — `analysis/FFM_2_1_latent_structure.py`.** Still not run. `FFM_2_4` measures
an effective dimension on its own 52-subject subset (75 of a 103 ceiling at ear level; 26
principal components at subject level for the MLP target), but that figure is a symptom of
the sample size — 104 points give a sample covariance with 103 non-zero eigenvalues — and
must **not** be quoted as the latent's intrinsic dimensionality. The script measures it
properly on all 713 subjects and remains the correct source.

**Earlier note that "the working scale is ~300 subjects, not 57".** True only for the HRTF
side and for meshes. It is **not** true for landmarks: landmark features are not poolable
across databases, so the landmark arm is capped at the 52 mapped CARLab subjects. Extending
to HUTUBS and SONICOM by landmark or anthropometric features would mean three independent
replications in three native feature spaces, compared on dCor²ᵤ — whose null depends only on
N and not on feature dimensionality — and not one pooled fit.

---

## 9B. Open actions *(supersedes §9)*

| # | Action | Blocks |
|---|---|---|
| 1 | **Download the HUTUBS and SONICOM meshes** (~258 subjects, both public, no permission needed) | The entire mesh arm; the only route to N ≈ 300 |
| 2 | **Request the SYMARE `Meshes/` tree from Prof. Jin / the PhD student**, and ask whether SYMARE v1.0 (Burton–Miller, 20 kHz) ever shipped — the v0.9 readme of 2014-03-20 says those simulations were in progress | The remaining 61 subjects; any BEM claim above 16 kHz |
| 3 | Run `analysis/FFM_2_1_latent_structure.py` on the full 713 subjects | The intrinsic latent dimensionality; carried over from §9 |
| 4 | Check which BEM set produced "BEM vs measured 6.28 dB, R² = 0.54" — `HE16kHz` (valid to 16 kHz) or `HTE05kHz` (valid only to **5 kHz**) | The validity of that figure above 5 kHz |
| 5 | Request the full `HRIRs/AcousticRaw/` archive from Prof. Jin | Any claim above 16 kHz; carried over from §9 |
| 6 | Learning curve: cross-validated score against subject count, on whichever geometry is obtained | Separates "geometry does not carry this" from "we never had enough people" — the one question N = 52 could not answer |
| 7 | The auto-decoding curve: freeze the decoder, fit `z` from N measured directions, sweep N = 3…100 | Converts R² = 0.05 into "geometry carries less than K measured directions", directly comparable to the few-shot tables in NIIRF / RANF / GraphNF-SCA |
| 8 | The repeatability ceiling from the HUTUBS repeat pairs (`pp22`/`pp88`, `pp1`/`pp96`) | Answers "the measurements are not repeatable enough for anything to be predictable" |
| 9 | Integrate `oslo_fast.py` | ~4× training throughput; carried over from §9 |
| 10 | Assert one direction grid per database in `preprocess_multidataset_stps.py` | Validity of the shared STPS matrix; carried over from §9 |
| 11 | Repository stays **private**. `results/ha_to_carlab_map.csv` and `results/symare61_to_carlab_RESOLVED.csv` carry real participant surnames; the removal procedure is in the repository README | Research-ethics compliance; carried over from §9 |

---

# Appended 2026-10-01 — Unit B work, 4 September to 1 October

## 2C. Dated work log *(continuation of §2B)*

### 2026-09-04 — The cohort question answered; the mapping retired; a four-database retrain

**Prof. Jin confirmed that the newly collected CAR Lab subjects and the 61 SYMARE subjects
are different people.** `symare61_to_carlab_RESOLVED.csv` had paired each SYMARE subject
with the acoustically nearest CAR Lab subject, which is a similarity-preserving map, so
every `landmark2z` number computed through it (dCor²ᵤ 0.071 at N = 52, CV-R² 0.052, the
4.208 dB end-to-end figure) is uninterpretable. They are retained in
`notes/landmark2z/RESULTS_landmark2z.md` as a documented error (§5C).

The argument recorded on 09-03 — that band-localised signal proved the mapping right — is
also withdrawn. A nearest-neighbour pairing preserves geometric similarity in the pinna
band, so band structure survives it; it is not evidence of correct pairing.

**Response: `FFM_2_6_QuadDB.ipynb`, retrained from scratch on four databases** (CAR Lab +
HUTUBS + SONICOM + SYMARE61 = **774 subjects**, 100 epochs). SYMARE enters as dataset
`id = 3` with a (0.60, 0.15, 0.25) split, like HUTUBS and for the same reason: it carries
geometry, so its test partition is reserved for downstream work. `split_v21.json` is reused
— `build_split` is incremental, so the CAR Lab, HUTUBS and SONICOM partitions and the frozen
CAR Lab test list are unchanged. FiLM β is initialised from the measured per-database mean
spectra (full-band mean level: CAR Lab −0.47 dB, SYMARE61 −5.74 dB). The point of the
retrain is not the extra 61 subjects (+8.6%) but that **SYMARE subjects now receive their
own latents and the mapping table leaves the pipeline entirely.** Checkpoints are
`FFM26_*`; `FFM23_*` is untouched.

Prior art obtained and read the same day: Martinez et al. (ICASSP 2026), Xiao et al. (ICASSP 2026)
and Zheng et al. (ICASSP 2026). SuDaField (Masuyama et al., IEEE OJSP 2025) had been read earlier,
at the end of Semester 1, sent by the supervisor; the Unit A report already discusses it
(corrected 10-22, §5C #26). SuDaField publishes per-dataset conditioning with code, so
cross-database FiLM is no longer claimed as a contribution (§8C).

### 2026-09-05 — FFM26 results; the landmark study redone on correctly matched subjects

**FFM26 validation**, epoch 99, pooled 1.4855 dB:

| Partition | Subjects | Val RMSE | Baseline | Reduction |
|---|---:|---:|---:|---:|
| CAR Lab | 218 | 1.3483 | 5.053 | 73.3% |
| HUTUBS | 96 | 1.8345 | 5.162 | 64.5% |
| SONICOM | 399 | 1.4545 | 4.977 | 70.8% |
| SYMARE | 61 | 1.3261 | 4.805 | 72.4% |

Adding SYMARE moves the three existing partitions by −0.024, +0.022 and +0.001 dB — all
below the 0.19 dB validation resolution. The defensible claim remains that pooling degrades
no partition, not that it improves any. v2.6 figures are not comparable with v2.3's
1.4867 / 1.2909; one model is to be used throughout the write-up.

**Corrected `landmark2z`** (`FFM_2_8`, `FFM_2_9`): 61 SYMARE subjects with their own
latents, 19 landmarks (13 ipsilateral pinna, 5 midsagittal, 1 shoulder), 46 development /
15 reserved test, fold-local PCA and standardisation throughout.

- Dependence: dCor²ᵤ = **0.0710**, permutation p = **0.0020**; controls clean (shuffled
  −0.0083, random features −0.0309).
- Usable signal: best development CV-R² **0.0845** (raw XYZ + per-landmark RFF + ridge);
  raw XYZ alone 0.0762; global RFF 0.0038. The RFF gain (+0.006) is within CV variability.
- MLPs memorise (train R² ≈ 0.98) and generalise below the mean predictor (−0.13 to −0.33).
- RFF scale: CV prefers smooth encodings (best near 1/σ = 200 mm); below 5 mm the score is
  ≈ 0. Nineteen landmarks do not carry millimetre-scale surface detail, so a fine-scale
  encoding of them has nothing to encode.
- **Stopping rule triggered** (0.0845 < 0.10). The 15 reserved subjects remain unread.

Repository: the invalid correspondence removed, corrected study and results document
committed (`docs/landmark2z/RESULTS_RFF_2026-09-05.md`).

### 2026-09-06 — Capacity diagnostics: neither longer training nor more features helps

Three follow-up questions, each kept separate from the headline table:

- **5000 epochs.** The MLP fits shuffled targets as well as real ones (train R² 0.990 vs
  0.988), and its held-out R² reaches only +0.008 at the pre-specified endpoint.
- **More Fourier features.** Ridge over M = 8 … 512: scores 0.056–0.093 with no monotone
  trend; the best (+0.0932 at M = 512) is +0.009 above the primary setting from one RFF seed.
- **More subjects.** An internal learning curve (8 → 31 training subjects) is not flat
  (slope ≈ +0.004 R² per subject), so N = 61 could not rule out a data-limited regime — the
  question the mesh arm was designed to answer (§2C, 10-01).

### 2026-09-09 to 09-11 — Results note for the supervisor; release staging

`landmark2z/README_supervisor.md` (403 lines) written for Prof. Jin: every statistic defined
before it is used (CV-R², unbiased dCor², permutation p), the data objects and partitions,
the main table, the capacity diagnostics and a bounded conclusion stating what the result
does and does not license. Learning-curve figure `results/landmark2z/FFM30_learning_curve.png`.
`FFM26_HEALPix_release_staging/` prepared for the full per-ear 19-band error maps, with the
rule that the guarded test set is evaluated once and only once. README and DATA.md updated
to FFM26; `FFM_3_0_CapacityStress.ipynb` saved with outputs in `landmark2z/` (not yet
committed).

### 2026-09-25 — Data-pipeline audit; literature survey

**Prompted by the supervisor's concern that `preprocess_multidataset_stps.py` builds one
interpolation matrix per database from the first subject's direction grid** (carried as
§9B action 10). Checked directly against every SOFA file:

| Database | Files | Directions | Max angular deviation from subject 1 |
|---|---:|---:|---:|
| HUTUBS measured | 96/96 | 440 | **0°** |
| SONICOM FreeFieldCompMinPhase | 399/399 | 793 | **1.5 × 10⁻⁶°** (float rounding) |

SYMARE uses one external `azim`/`elev` grid by construction and CAR Lab caches one matrix
per grid. **The shortcut is a code-quality issue, not an error in any exported array.**
Two genuine issues remain and are recorded instead:

1. **The lower hemisphere is extrapolated for three of four databases.** SONICOM, SYMARE and
   CAR Lab stop at −45° elevation; below that is (1 − sin 45°)/2 = **14.6%** of the HEALPix
   sphere (~450 of 3072 pixels), covering 678 of 774 subjects. No direction mask was found
   in the analysis scripts.
2. **Processing conventions differ.** SONICOM enters free-field compensated (minimum-phase
   EQ); HUTUBS enters as the measured SOFA. Part of what FiLM absorbs is this difference.

Also found: **HUTUBS ships BEM-simulated HRIRs for all 96 subjects** (`pp*_HRIRs_simulated.sofa`,
1730 directions) alongside the measured ones — a ready-made simulated/measured pair for
Thread A that had not been used. Literature survey of ICASSP 2021–2026 HRTF work compiled
into a prioritised reading list (DNN-augmented BEM 2021, predictability from ear shapes
2021, HRTF Field 2023, measurement-setup differences 2023, NIIRF 2024, RANF 2025, diffusion
2025, Martinez / Xiao / Zheng 2026), plus the EUSIPCO 2025 boundary-integral network and the
2025 OJSP survey.

### 2026-10-01 — Mesh → latent on 200 SONICOM meshes, run to its stopping rule

The question §9B left open — is the landmark result a ceiling of geometry or of N = 61? —
answered with dense meshes at N = 164.

**Data.** The extended SONICOM release provides, for the 200 subjects whose scans passed the
curators' quality control, a pre-processed head mesh, ipsilaterally graded meshes and a
Mesh2HRTF simulation on the measured 793-direction grid. Downloaded (≈4.8 GB). Audited on
the files themselves: millimetres; origin at the interaural midpoint (ear-canal positions
from the simulation's `ReceiverPosition`); +x forward, +y left, +z up; 375k–778k triangles,
so all sampling is by area. **24 of 200 "watertight" meshes have open or non-manifold edges**
— none within 13 mm of an ear canal, at most 5.2 mm of edge inside the ear crop. Neck
truncation height varies from −148 to −81 mm. The `*_Scaled.sofa` pair was checked against
the local files: the synthetic one is the Mesh2HRTF output minus a per-subject constant
(direction-dependent residual ≤ 0.10 dB); the measured one carries a shaped mean-spectrum
difference from the free-field-compensated measurement (−2.4 … +9.4 dB) and is not used.

**Sampling** (`mesh2z/sample_meshes.py`): 50k area-uniform head points and 20k points within
45 mm of each ear canal, right ear mirrored into the left-ear frame. Every subject passes ten
loud checks (scale, closure near the ears, outward normals, midline nose, nose height,
crown, front/back, ear position and symmetry, acoustic laterality, ear canal on the surface).
Six deliberate corruptions of real meshes — metres, inverted normals, a hole at the ear
canal, a 20 mm shift, a 180° turn, unchanged — fail and pass as intended. **200/200 pass.**

**Targets and reference levels** (`mesh2z/FFM_3_1_Mesh2Z_Latents.ipynb`): all 399 SONICOM
subjects encoded with FFM26 (inverse FiLM before the encoder, asserted bit-identical to
`model.forward`). Meshed subjects by AE partition: 147 train / 17 val / **36 test — encoded
but never scored, reserved as the final mesh2z test.** On the 164 development subjects
(pooled RMSE, DC–15.85 kHz / 6–12 kHz):

| Reference | All | 6–12 kHz |
|---|---:|---:|
| ① AE reconstruction | 1.396 | 1.655 |
| ② Both ears share one latent (subject average) | 2.601 | 3.026 |
| ③ Decoded population-mean latent | 4.844 | 5.587 |
| ④ Population-mean HRTF | 4.842 | 5.578 |

The val subset reproduces the recorded SONICOM validation figure (1.452 vs 1.4545 dB).
**Averaging the two ears costs 1.2 dB** — same-subject latent cosine is only 0.527 — so the
regression was made per ear, with cross-validation grouped by subject. The meshed subset is
mildly less atypical than the rest (latent norm median −3%, Mann–Whitney p = 0.030).

**Regression** (`mesh2z/FFM_3_2_1_Mesh2Z_RFF.ipynb`): random-Fourier-feature mean embeddings of
the ear crop (canal-relative) and of the head above z = −75 mm, σ ∈ {0.005 … 0.1} mm⁻¹,
M = 256; nine simple geometric measures as the comparator; kernel ridge with α chosen by
inner grouped CV; 10 outer folds. Self-checks: random features CV-R² −0.0001; the retrieval
metric identifies 100% of subjects from their own AE reconstruction.

| Configuration | CV-R²(z) | End-to-end dB | 6–12 kHz | Gap recovered | Rank-1 identification |
|---|---:|---:|---:|---:|---:|
| ④ Population mean | — | 4.857 | 5.591 | 0% | — |
| Simple measures | 0.043 | 4.711 | 5.487 | 4.2% | 2.1% |
| Ear RFF, multi-σ | 0.067 | 4.657 | 5.374 | 5.8% | 3.0% |
| Head RFF, multi-σ | 0.026 | 4.764 | 5.540 | 2.7% | 1.5% |
| **Simple + ear RFF** (best of 15) | **0.076** | **4.631** | **5.370** | **6.5%** | **2.1%** |
| ① AE reconstruction | — | 1.396 | 1.655 | 100% | — |

Chance identification is 0.6% (1 in 164). Max-statistic permutation test over all 15
configurations: observed 0.0757 against a null 95th percentile of 0.0014, **p = 0.010**
(the floor for 100 permutations). Learning curve: 0.038 / 0.053 / 0.069 / 0.077 at
N = 41 / 82 / 123 / 164.

**Where the signal is.** Per-frequency CV-R² peaks at ~0.21 near 5.2 kHz and falls to
0.04–0.05 above 10 kHz; the end-to-end gain over the mean is largest at 5.2 kHz (0.47 dB),
~0.3 dB at 6–9 kHz and ~0.1 dB at 11 kHz. The ear crop carries more than the head, and the
best single scales are 1/σ = 20–50 mm, at which a 1.5 mm registration error changes the
features by only 0.26–0.33 of the between-subject distance.

**Conclusion.** Dense mesh geometry carries information about an individual's HRTF that is
statistically unambiguous and practically small: 0.23 dB, 6.5% of the gap between the
population mean and the autoencoder, and concentrated at 0–6 kHz rather than in the
pinna-notch band. The curve is still rising at N = 164, but with diminishing increments.
**Stopping rule triggered (0.076 < 0.10); the mesh → latent arm is closed.** It does not
show that geometry cannot determine the HRTF — a BEM solve from the same mesh does exactly
that — only that a linear map from global shape descriptors to this latent recovers little
of it. That is the measurable form of Thread A's premise: notch structure has to come from
physics on the surface, not from a regression on shape summaries.

### 2026-10-02 — Exploratory: does a trained point network beat random features?

**Status: exploratory, after the stopping rule had triggered.** Recorded because it answers
a different question — learned versus random point features — not to revise the 10-01
conclusion. `mesh2z/FFM_3_3_Mesh2Z_PointNet.ipynb`; the same 164 subjects and the **same
outer folds** as `FFM_3_2` (exported as `oof_ridge.npz`; the notebook recomputes ridge's
CV-R² from the exported predictions and asserts agreement with `FFM_3_2`). Ear point cloud
(canal-relative, right ear mirrored), shared per-point MLP (64-128-256), max + mean pooling,
MLP head to the top 64 fold-local principal components of z; ~0.2 M parameters. Inputs
either raw xyz or RFF (σ = 0.01/0.02/0.05 mm⁻¹). 200 epochs, architecture and optimiser
fixed in advance; a shuffled-target control trained in every fold; curves logged for
inspection only.

`FFM_3_2` was rerun first to add LSD and export its out-of-fold predictions; CV-R² (0.0757)
and the max-statistic p (0.0099) reproduced exactly.

| | CV-R²(z) | Pooled dB | LSD | LSD, el ≥ −45° | Paired vs best ridge (95% CI) |
|---|---:|---:|---:|---:|---|
| ④ Population mean | — | 4.857 | 4.646 | 4.603 | |
| Ridge, simple + ear RFF | +0.076 | 4.631 | 4.398 | 4.348 | reference |
| Ridge, simple measures | +0.043 | 4.711 | 4.477 | 4.428 | |
| PointNet, xyz | +0.023 | 4.774 | 4.537 | 4.489 | **+0.142 dB [+0.105, +0.180]** |
| PointNet, RFF | −0.121 | 5.144 | 4.876 | 4.813 | **+0.512 dB [+0.425, +0.597]** |
| ① AE reconstruction | — | 1.396 | 1.228 | 1.335 | |

Shuffled controls: xyz −0.024, RFF −0.292. The RFF network fits its training targets to
R² 0.71 and fits **shuffled** targets almost as well (0.67), while held-out R² falls from
about +0.03 near epoch 30 to −0.12 — memorisation. The xyz network sits above its control
(+0.023 vs −0.024) but below nine hand-made measures (+0.043). Even the best point of either
curve, which the protocol does not select, stays below ridge. **At N = 164 learned point
features are worse than random features with strong shrinkage**, by margins whose
bootstrap intervals exclude zero. This is the sample-size signature already seen with
landmarks, not evidence about geometry.

**LSD.** LSD here is RMS over the 93 bins (DC–15.85 kHz) per direction, then the arithmetic
mean over directions and ears — the literature convention, always below the pooled RMSE.
On SONICOM the **population mean alone scores 4.65 dB LSD**, inside the 3.8–5.1 dB range
that recent geometry- or anthropometry-driven papers report as method results (Zhao 2024,
Niu 2025, Xiao 2026) without a population-mean baseline. Databases, bands and grids
differ, so this is not a ranking — it is the reason every figure in this project travels
with its baseline.

### 2026-10-02 (later) — Why mesh → latent is weak: the physics ceiling

`mesh2z/physics_ceiling.py` (two steps: `decode` needs torch, `analyse` needs h5py; they
segfault together in one local environment). The question it answers: **does an exact
acoustic solve from the same mesh predict the person's measured HRTF better than the
population mean?** If not, no regressor on these meshes can be expected to. SONICOM's
Mesh2HRTF simulations (same 793-direction grid, asserted for all 164 development subjects)
were pushed through the project's own STPS → HEALPix pipeline, which is asserted to
reproduce the stored measured arrays to 0.0 dB. The 36 locked subjects were not read.

Comparisons use the **DTF** (each ear minus its own direction-mean spectrum, which removes
level and calibration so a simulation and a measurement can be compared) on the **measured
sphere** (elevation ≥ −45°); pooled RMSE over 164 subjects × 2 ears × 93 bins × pixels.

| Predictor of each ear's measured DTF | RMS dB | 6–12 kHz | R² vs mean | Rank-1 identification | Median rank |
|---|---:|---:|---:|---:|---:|
| Population mean (leave-one-out) | 3.81 | 4.57 | 0 | — | — |
| **Ridge** (mesh features → z → decoder, FFM_3_2_1) | **3.70** | **4.41** | **+0.060** | 1.8% | 17 |
| Own Mesh2HRTF simulation, sim-vs-measurement bias removed (LOO) | 4.67 | 5.46 | **−0.500** | **11.9%** | 16 |
| Same, optimally shrunk towards the mean (β = 0.19, LOO) | 3.76 | 4.48 | +0.029 | 11.9% | 20 |

Chance identification is 0.6% (median rank 82). Correlation between (simulation − mean) and
(ridge − mean): +0.18.

**Reading.** The exact BEM solve from a person's own scan is *further* from their measured
HRTF than the population mean is. It does carry person-specific structure — it picks the
right person out of 164 twenty times more often than chance — but only about a fifth of its
deviation from the mean (β = 0.19) is borne out in the measurement, and even optimally
shrunk it explains half the variance the ridge does. **The ceiling on mesh → measured HRTF
for these scans is set by the scan–measurement relationship, not by the regressor.**

Two supporting measurements on the same subjects:

- **Direction-independent variation.** 35.8% of between-person HRTF variance is a
  direction-independent spectrum per ear; below 0.5 kHz it is **96.5%**. Head and pinna
  are acoustically small there, so that 1.9 dB spread is predominantly measurement
  (microphone placement, calibration), not anatomy — irreducible from geometry.
- **The two ears of one person** differ by 4.07 dB DTF (4.79 dB at 6–12 kHz) against
  5.41 dB (6.52 dB) between different people. Real asymmetry and per-ear measurement
  variability cannot be separated here; both bound any geometry-based predictor.

**Reconciling 3.70 with 4.631 dB.** Same ridge predictions, four conventions (mean is the
leave-one-out mean over these 164 subjects, hence 4.797 rather than FFM_3_2_1's 4.857 from
the 339-subject pool): full HRTF on all 3072 directions 4.797 → 4.631 (reproducing
FFM_3_2_1 exactly); measured sphere 4.758 → 4.583; DTF on all directions 3.884 → 3.765; DTF
on the measured sphere 3.814 → 3.697. Removing the direction-independent spectrum accounts
for ~0.9 dB of the difference; the gain from geometry is 0.12–0.18 dB (~3%) in every one.

**Caveats.** One global bias correction and one global shrinkage factor, both leave-one-out;
per-frequency calibration could improve the simulation rows somewhat, but not from −0.50 to
above the ridge. SONICOM's simulations omit the torso (scans stop at the neck), which
accounts for some low-frequency mismatch. Scans and measurements were made in separate
sessions. N = 164 quality-controlled subjects.

**Consequences.** (i) For the thesis, the mesh → latent bound becomes a statement about the
data rather than the model. (ii) For Thread A, a surrogate trained on BEM output inherits
the BEM–measurement gap: on either database its natural target is the simulation (speed
and fidelity to BEM), not the measurement. (iii) **Not a property of optical scans.** The same question asked of SYMARE — MRI-derived
meshes, BEM computed by the database authors (`Bem/HE16kHz`), N = 61 — with identical code
on each database's native grid (`mesh2z/sim_vs_measured.py`), each side compared with its own
population mean:

| 0.2–16 kHz, per side | r(dev_measured, dev_BEM) | 6–12 kHz r | BEM as predictor, R² | Own simulation closest |
|---|---:|---:|---:|---:|
| SYMARE (MRI) | +0.225 | +0.293 | −0.524 | 29.5% (chance 1.6%) |
| SONICOM (optical) | +0.164 | +0.161 | −0.532 | 8.8% (chance 0.6%) |

Both databases show the same picture: the BEM from a person's own mesh carries their identity
well above chance but correlates only ~0.2 with how they actually differ from the population,
and used as a predictor it is worse than the mean. MRI meshes do not remove the gap. This
agrees with Zhang, Wang & James (ICASSP 2021), who found BEM "converges to a wrong
distribution" on HUTUBS. An earlier draft of this entry claimed a four-fold SYMARE advantage;
that compared a side-mixed SYMARE figure with a per-side SONICOM one (§5C #21).

### 2026-10-04 — Within one person: does a change in the pinna predict a change in z?

`mesh2z/within_person.py`. Prompted by the question of whether a model could learn "how a change
in the pinna changes z" and so produce a reliable z for a new pinna. Between people that is the
same model as the existing ridge (z_k − z_i = W(s_k − s_i) is z = Ws + b; 164 people give
~13 000 pairs but still 164 independent observations, and the part of each HRTF that geometry
cannot reach adds rather than cancels). **Within one person** it is a different and cleaner
question: the two ears share head, torso, age, session and rig, so a left–right shape
difference can predict a left–right latent difference only through the pinna (or through what
differs between the two ears' measurements).

Same ear-crop features as `FFM_3_2_1` (seven per-ear simple measures; ear RFF, five σ, M = 256),
right ear mirrored, one row per subject, kernel ridge with inner CV, 10-fold outer CV, fold-local
target PCA; self-check with random features −0.0000.

| | Within person (left vs right) as a fraction of between person |
|---|---:|
| Pinna geometry, simple measures | 0.58 |
| Pinna geometry, ear RFF | 0.66 |
| Latent z | 0.67 |

| Predict z_L − z_R from pinna_L − pinna_R | CV-R² |
|---|---:|
| Simple measures | +0.0030 |
| Ear RFF, multi-σ | +0.0046 |
| Both | +0.0046 |

Max-statistic permutation over the three (200): null 95th percentile +0.0003, **p = 0.005**
(the floor for 200).

**Reading.** The two ears of one person differ in shape by about 60% as much as two different
people, and their latents by 67% — yet the shape difference explains **< 0.5%** of the latent
difference. The pinna → z relationship is real once every person-level factor is removed, and
almost none of an individual ear's latent is reachable from these pinna descriptors. Set beside
the between-person 0.076, most of that figure is likely carried by person-level size and shape,
not by the pinna detail that differs between one person's ears. Consistent with the physics
ceiling (per-side BEM r ≈ 0.2): either the descriptors (and the BEM) miss what matters, or the
per-ear measurement varies by about as much as the pinna does. SONICOM has no repeated
measurement of the same ear, so the two cannot be separated here.

**Consequence.** A pairwise or "change → change" model is not a route to reliable z with these
data. What would change the picture is data — repeated measurements of the same ear, to size
the measurement term — not another model.

### 2026-10-05 — Closing experiment: pinna only, pinna alignment, and neighbours

`mesh2z/FFM_3_4_Mesh2Z_PinnaOnly.ipynb` (written 10-04, run on Colab 10-05; outputs in
`mesh2z/output/mesh2z_pinna_v1/`). Two remaining intuitions tested together, under a rule
fixed before the run: *compare the best pinna-only configuration with the FFM_3_2_1 ridge on
the same folds, subject by subject; if the 95% bootstrap interval of the end-to-end gain
includes 0 or lies on the wrong side, mesh → latent is closed, whatever the result.*

**A. Pinna only.** Only points protruding > 3 mm from the canal plane (median 62% of the 45 mm
crop; 4096 per ear), optionally aligned to a template pinna by rigid ICP without scaling
(template iterated three times on the development set; rotation median 9°, residual 2.4–2.8
mm). Same folds, kernel ridge, fold-local target PCA:

| Features | CV-R²(z) |
|---|---:|
| Pinna RFF, canal frame | **+0.058** |
| Pinna RFF, ICP-aligned | +0.044 |
| Point-to-point correspondence after ICP | +0.039 |
| Seven ear measures | +0.044 |
| Measures + aligned RFF / + correspondence | +0.054 / +0.055 |
| *Reference: FFM_3_2_1 ridge (45 mm crop + measures incl. head)* | *+0.076* |

Max-statistic permutation over the six: p = 0.005 (null 95th percentile +0.0011). Restricting
to the pinna loses signal, and aligning the pinna loses more: part of what the geometry
carries is where and how the pinna sits on the head (protrusion, orientation) and the
surrounding head surface — exactly what pinna alignment removes.

**B. Are shape neighbours z neighbours?** No regression; per side, then averaged.

| Shape description | Spearman of pairwise distances | Shape-nearest k also z-nearest: k = 5 / 10 / 20 (chance 3.1 / 6.1 / 12.3%) |
|---|---:|---|
| Point correspondence | 0.137 (p = 0.003) | 7.4% / 11.2% / 18.1% (each p = 0.005) |
| Seven ear measures | 0.176 (p = 0.001) | 5.7% / 10.8% / 18.2% (each p = 0.005) |

Pinna shape PC1 against latent PC1 (left ears): Spearman +0.04. Shape neighbours are z
neighbours about twice as often as chance, but most are not; a one-dimensional ordering by
shape shows no ordering in z.

**C. End to end.** Best pinna-only (pinna RFF, canal frame) 4.685 dB pooled / 4.454 dB LSD
against the FFM_3_2_1 ridge's 4.631 / 4.398. Paired over subjects: **+0.054 dB worse,
95% CI [+0.029, +0.079]**. By the rule fixed in advance, **mesh → latent is closed.**

**Where the arm ends.** Landmarks (N = 61), meshes (N = 164), a trained point network, the pinna
alone, pinna alignment and a within-person design all agree: geometry carries a statistically
unambiguous but practically small amount of information about an individual's measured
latent (best 6.5% of the gap to the autoencoder). An exact BEM solve from the same mesh does
no better against measurements, on MRI or optical meshes. The open question is the size of
the per-ear measurement term, which only repeated measurements can answer (§9C action 4c).

### 2026-10-05 (later) — The same question on a fixed train / val / test split

`mesh2z/FFM_3_5_Geometry2Z_TrainValTest.ipynb` (written and run on Colab 10-05). Purpose: check
that the cross-validated conclusions are not an artefact of cross-validation, using one fixed
split, MLPs trained epoch by epoch with checkpoints, and the same controls on both tasks. Not a
new selection: the input features are the FFM_2_9 (landmarks) and FFM_3_2_1 (meshes)
configurations, and the network settings were fixed before the runs below. A first,
interrupted mesh run (constant learning rate, dropout 0.3) was discarded because its training
loss stopped falling; cosine decay to 1e-5, gradient clipping at 1.0 and dropout 0.1 were set
before these runs.

Validation subjects are drawn at random from the development set (split seed 0); the reserved
test subjects are the same as before and were not read. One MLP seed (0). The MLP's best epoch
and ridge's α are both chosen on the validation set, so those two rows are optimistic.

| Validation, pooled dB (recovered %) | Landmarks, SYMARE — 37 / **9** / 15 | Meshes, SONICOM — 131 / **33** / 36 |
|---|---:|---:|
| Population mean (training ears) | 4.800 | 4.796 |
| Ridge, α chosen on validation | 4.773 (0.7%) | **4.715 (2.4%)**, identification 7.6% vs 3.0% chance |
| MLP, best validation epoch | 4.849 (−1.4%), epoch 283 | 4.834 (−1.1%), epoch 16 |
| MLP on shuffled targets | 4.878 (−2.1%) | 4.903 (−3.1%) |
| Autoencoder reconstruction | 1.150 | 1.403 |

- **The MLP is worse than the population mean on both tasks**, even at the epoch chosen on the
  validation set, and within 0.03–0.07 dB of the shuffled-target control. On meshes the
  validation error is lowest at epoch 16 and rises from 4.83 to 5.17 dB while training R²
  climbs to 0.85; on shuffled targets training R² still reaches 0.70 (landmarks: 0.996).
- **Ridge is the only row better than the mean**: 2.4% of the gap on meshes; 0.7% on
  landmarks, which nine subjects cannot distinguish from zero.
- *Not like for like with FFM_3_2_1.* That notebook weighted each feature block equally (the nine
  simple measures as much as each 512-column RFF block); FFM_3_5 standardises all 2569 columns
  together, so the simple measures carry under 1% of the weight. This, and a 33-subject
  validation set in place of 164-subject cross-validation, may account for 2.4% against 6.5%;
  neither has been tested.

**Consequence.** None for the conclusions: a fixed split reproduces the cross-validated result,
and the network does not extract more than a linear model. FFM_3_5's test cell would use up the
same 36 reserved mesh subjects as §9C action 3 (see 3a).

### 2026-10-06 — Fidelity audit of the mesh inputs; complete meshes packed; FFM_3_6

**Audit** (`notes/mesh2z_audit_2026-10-06/AUDIT_CN.md`, outside the repository). The sampling
chain was replayed from the source STL and SOFA files for all 200 subjects: source ↔ sample
correspondence, point and normal arrays, mirroring, canal positions, crop areas and the uploaded
zip all match bit for bit. No conversion error. It found four reasons why "mesh → latent fails"
could not yet be said of the *complete* mesh:

1. The inputs were 20 000 random points per ear. Re-sampling the same mesh moves the RFF feature
   by 12–23% of the typical between-person distance (σ = 0.005 → 0.1 mm⁻¹).
2. FFM_3_5's MLP predicted only the first 64 target PCs (82% of training variance; projecting the
   true validation z onto them gives R² 0.63), while ridge predicted the full z — so "MLP vs
   ridge" was not like for like (§5C #22).
3. FFM_3_5's feature weighting differs from FFM_3_2_1 (already noted, 10-05 later).
4. The local cache in Colab was reused on sight of a manifest; nothing checked file identity.

It also states what cannot be fixed here: the source is SONICOM's *preprocessed* mesh (smoothed,
hair removed; the raw 3DSCAN is not held), and the canal positions come from the simulation's
receivers, not an independent calibration.

**Complete meshes packed** (`mesh2z/pack_full_meshes.py`): the 200 `*_preprocessed.stl` files,
unmodified (4.87 GB; 2.96 GB zipped), with a manifest of per-file SHA-256, canal positions and the
crop areas of `sample_meshes.py`. Each STL's SHA-1 equals the `stl_sha1` recorded in the samples
the earlier notebooks used; canals equal the samples' manifest; every file is read back from the
zip and checked. The zip is deterministic (re-packing gives the same SHA-256,
`a73401a1…124f8d`).

**FFM_3_6** (`mesh2z/FFM_3_6_Mesh2Z_FullMesh.ipynb`, written 10-06, not yet run on Colab):
- Features are computed from every triangle of the ear crop, with no sampling: the RFF mean
  embedding by an exact closed-form integral per triangle (second divided difference of
  e^{it}, Hermite–Genocchi), and the nine geometric measures as exact quantiles of the surface-area
  measure (each triangle contributes a triangular distribution; half-space cuts at y = 3 mm and
  z = −75 mm are exact). Same crop, same frequencies and same split as FFM_3_5.
- MLP and ridge share the output space (all training PCs) and the FFM_3_2_1 block weighting;
  an oracle row shows what that output space allows.
- Pre-declared comparisons: exact vs the same features from 20 000 random points (three seeds);
  block vs column weighting; M = 256 vs 1024; with vs without surface normals; MLP vs ridge.
  Stopping rule unchanged (all validation R²(z) < 0.10 → conclusion stands, now for complete-mesh
  inputs).
- Every STL is checked against its SHA-256 when read; the zip against the hash in Cell 0; the
  feature cache is named by a content fingerprint. FFM_3_5 and FFM_3_6 share one test-spent marker.
- Checks before any target is read: closed form vs 80 × 80 Gauss quadrature (max error 4 × 10⁻¹⁰
  of the triangle area, including degenerate triangles); crop area equal to `sample_meshes.py` to
  10⁻⁹; mirrored right-ear normals outward; cut areas conserved; GPU float32 vs float64 5 × 10⁻⁸;
  Monte Carlo error falls as 1/√N towards the exact value (6.3 × 10⁻³ at 20 000 points).

### 2026-10-06 (later) — Complete meshes: FFM_3_6 and FFM_3_7 results

Both run on Colab on the same 131 / 33 split as FFM_3_5 (checked subject by subject against the
3_5 checkpoint); the 36 locked subjects were not scored. Outputs are saved in the notebooks and in
`mesh2z/output/mesh2z_fullmesh_v1/`, `mesh2z/output/mesh2z_direct_graph_rff_v1/`.

**FFM_3_6 — exact features from every triangle, ridge and MLP in the same output space.**
Features for 200 subjects took 164 s. All self-checks passed as locally.

| Validation (33 subjects, 66 ears) | R²(z) | pooled dB | recovered | identification (chance 3.0%) |
|---|---:|---:|---:|---:|
| Population mean (training ears) | 0 | 4.796 | — | 3.0% |
| Ridge, exact, block-weighted (primary) | +0.063 | **4.649** | **4.3%** | 9.1% |
| Ridge, the same from 20 000 random points (3 seeds) | +0.062 … +0.063 | 4.647 – 4.652 | 4.2 – 4.4% | 9.1% |
| Ridge, exact, column-weighted (= FFM_3_5) | +0.043 | 4.714 | 2.4% | 7.6% |
| Ridge, exact, M = 1024 | +0.064 | 4.647 | 4.4% | 9.1% |
| Ridge, exact + surface normals | +0.059 | 4.669 | 3.8% | 9.1% |
| MLP, exact, all 261 PCs | +0.007 | 4.806 | −0.3% | 3.0% |
| MLP on shuffled targets | +0.001 | 4.813 | −0.5% | 3.0% |
| Oracle: true z on the 261 training PCs | +0.762 | 2.303 | 73.5% | 100% |
| Autoencoder reconstruction | 1 | 1.403 | 100% | 100% |

Pre-declared comparisons (Δ pooled dB, 95% CI by subject):
- **Sampling noise does not matter for prediction.** The random-point features differ from the
  exact ones by 10–23% of the between-person distance (σ = 0.005 → 0.1 mm⁻¹, reproducing the
  audit), yet ridge on exact vs random features: −0.003 [−0.009, +0.003], +0.002 [−0.005, +0.009],
  −0.002 [−0.011, +0.008].
- **The weighting explains FFM_3_5's ridge.** Column weighting reproduces 3_5's 4.715 dB (4.714);
  block weighting is better by 0.065 [0.031, 0.101].
- **A larger descriptor or surface orientation adds nothing.** M = 1024: +0.003 [−0.004, +0.009];
  adding normals makes it slightly worse (exact − normals −0.019 [−0.042, +0.002]).
- **The MLP is worse than ridge with the same output space and weighting**: +0.156
  [+0.079, +0.231]; against the mean +0.009 [−0.016, +0.035]. Its training R² stayed at 0 for
  about 140 epochs (predicting the mean), then rose to 0.45 while validation fell to 5.15 dB; the
  shuffled-target control never fitted at all.
- **Stopping rule triggered**: the best row (ridge, M = 1024) has R²(z) 0.064 < 0.10.

**FFM_3_7 — a learned network on every face of the whole head.** (`FFM_3_7_Mesh2Z_DirectFullMesh.ipynb`
with `fullmesh_graph.py`.) Every original triangle (157 000 – 905 000 per subject, no crop,
sampling or decimation) enters a shared corner MLP with per-vertex RFF (σ = 0.02, 0.1, 0.5,
2 rad/mm, 16 frequencies each), then three message-passing blocks over shared edges, area-weighted
and soft canal-centred readouts, and a head onto all 261 training PCs. 136 197 parameters; one ear
per forward pass, gradients accumulated over 4 ears; AdamW 3 × 10⁻⁴, cosine to 10⁻⁵ over 100
epochs; 23 s per epoch on an RTX PRO 6000. Shuffled-target control not run.

| Epoch | train optimisation loss | train probe R² (8 subjects) | val R²(z) | val dB |
|---:|---:|---:|---:|---:|
| 1 | 1.008 | −0.010 | −0.003 | 4.814 |
| 24 (best) | 0.883 | +0.070 | **+0.044** | **4.671** |
| 44 | 0.764 | +0.222 | −0.005 | 4.820 |
| 100 | 0.636 | +0.359 | −0.035 | 4.894 |

- Best validation: 4.671 dB, **3.7% of the gap**, R²(z) 0.044; against the mean −0.126 dB
  [−0.254, +0.010] — the interval includes 0 and does not account for choosing the best of 100
  epochs on these same 33 subjects. Below ridge on exact global features (4.649 dB).
- After epoch 24 the training fit keeps improving (probe R² 0.07 → 0.36) while validation falls
  is worse than the mean from about epoch 44. The flat training loss over the last 15 epochs is the
  learning rate reaching 10⁻⁵, not convergence.
- **No further training**: the generalisable part is learned by epoch ~24 and is ridge-sized;
  more epochs or a longer schedule add only training fit.

**Conclusion of the mesh arm, now for complete meshes.** With every triangle entering exactly,
matched output spaces and weighting, a larger descriptor, surface normals, an MLP and a learned
whole-mesh graph network, the best validation result is linear ridge at 4.3–4.4% of the gap
(R²(z) 0.06). Detectable, not usable — the conclusion of 10-01 and 10-05 stands, and it no longer
depends on point sampling. It still concerns SONICOM's preprocessed meshes, simulation-defined
canal positions, these representations and one 33-subject validation split.

### 2026-10-07 — Literature check and a proposed positioning *(tentative, not yet agreed)*

**Question.** Is there published work — ICASSP preferably — that states the geometry result, or
does the field avoid it?

**Answer.** No single paper states it: that geometry recovers only a small fraction of an
individual's *measured* HRTF beyond the population mean, and that an exact BEM solve from the
same scan does not reproduce individual deviations. The pieces exist separately, each with its
own metric, baseline and data, and the field's own survey and challenge document why they are
rarely put together.

| Evidence | What it shows | What it does not settle |
|---|---|---|
| Fantini et al. 2025, *IEEE OJSP* survey of 76 ML individualisation papers | Lists 3 studies with an average-HRTF baseline and 8 with significance tests; no standard evaluation protocol; its reference line is KEMAR (6.6 dB SD), not the population mean; "limitations of anthropometry-based methods" is its first open gap | — |
| LAP Challenge 2024 Task 2 (Hogg et al., *IEEE OJSP* 6, 2025) | With a shared benchmark, most submissions did not beat a generic HRTF averaged over the training set | The task is upsampling from sparse measurements, not geometry |
| Lee & Kim 2018, *Appl. Sci.* 8:2180 | By RMSE the average HRTF was best (their DNN +1.91 dB, proposed +0.88 dB); the claims rest on LSD and a 5-listener test | — |
| Zhang, Wang & James, ICASSP 2021 (HUTUBS, 58 subjects) | BEM converges with mesh refinement but keeps a discrepancy to measurement ("converges to a wrong distribution"); BEM SD 7.23 dB | Their "average" bias averages over frequency too; individual deviations not tested against the population mean |
| Brinkmann et al. 2019, *JAES* (HUTUBS) | Measured vs simulated: < ~1 dB up to 5 kHz, 7 dB on average at 17.6 kHz; simulations carry more high-frequency energy; audible (tone colour, ~12° upward). One repeated human measurement: < ~1 dB on average, up to 5 dB above 7 kHz (95th percentile) | No population-mean comparison |
| Jin/Guillon et al., ICME 2012 (SYMARE, 10 listeners) | Effective head diameter, measured vs FM-BEM, r² = 0.64; optimal frequency scaling of a few per cent; main spatial features present | Structural agreement only; individual deviations not compared with the mean (consistent with §2C 10-02: per-side r ≈ 0.2) |
| Andreopoulou, Begault & Katz 2015, *IEEE JSTSP* | One dummy head, 12 sets from 10 laboratories: differences up to 12.5 dB below 6 kHz and 23 dB above | Dummy head, not repositioned humans |
| Dinakaran et al., ICASSP 2018 | ~1 mm geometric precision needed for localisation cues; ~4 mm keeps the overall spectral shape | One dummy head |
| Zhou et al., ICASSP 2021 | HRTFs are predictable from ear shape | Targets are simulated from the same meshes: no measurement term |
| *Counter-evidence:* Ziegelwanger, Majdak & Kreuzer 2015, *JASA*; Poole & Picinali 2026, arXiv:2608.16722 (SONICOM, 200) | With 1 mm meshes, localisation with calculated HRTFs matches measured; on SONICOM, synthetic HRTFs are closer than KEMAR in ITD/ILD and match measured in a VR localisation test (N = 20) | Perceptual criterion; baselines KEMAR and one random subject; no population mean, no test of individual spectral deviations |
| Niu et al. (WASPAA 2025), Xiao et al. (ICASSP 2026), Zheng et al. (ICASSP 2026) | Anthropometry → HRTF, LSD 4.90–5.09, 4.611 and 3.71 dB | None reports the population-mean LSD on the same data (here it is 4.65 dB on SONICOM; not directly comparable) |

The counter-evidence and this project agree once the criterion is separated: what survives from
geometry or simulation is head size, ITD/ILD and the low-frequency spectral shape (this project's
geometry signal sits at 0–6 kHz); the 6–12 kHz pinna detail is where simulation and measurement
part in every source above. "Perceptually adequate" is a weaker criterion than "closer to this
person's measured spectrum than the population mean".

**Proposed positioning of the geometry chapter** *(to be decided with the supervisor; thesis
text unchanged)*: not a new method, but a baseline-referenced measurement on one scale — 0% = the
population mean, 100% = the autoencoder reconstruction — on subjects who have geometry,
simulation and measurement together.

- **Gap 2 (no population-mean baseline).** The mean alone is 4.86 dB pooled / 4.65 dB LSD; every
  result is reported as the fraction of the mean-to-autoencoder gap it recovers, with
  identification against chance.
- **Gap 3 (no measured bound on predictability from geometry).** Landmarks to complete-mesh graph
  networks: detectable (p ≤ 0.01) but at most 6.5% of the gap (0.23 dB; cross-validated) and
  4.3% on the FFM_3_6 validation split; stronger models do not help (MLP 0.156 dB worse than
  ridge), and exact integration over every triangle changes nothing (≤ 0.003 dB).
- **Gap 4 (BEM–measurement gap on individuals not quantified).** An exact BEM solve from a
  person's own scan predicts their measured individual deviation worse than the mean (DTF R²
  −0.50; +0.03 optimally shrunk, below ridge), although it identifies the person (11.9% vs 0.6%);
  per-side correlation of simulated and measured deviations ≈ 0.2 on MRI (SYMARE) and optical
  (SONICOM) meshes; within one person, the left–right pinna difference explains < 0.5% of the
  left–right latent difference.

**Wording limits if adopted.** Not "an upper bound on predictability from geometry" but "the
predictability measured with these representations, scans and measurement conditions"; not "the
limit is not the regressor" but "stronger regressors among those tested did not help"; the
criterion is spectral magnitude, not perception.

**Weak points to state.** No repeated measurements of the same ears, so measurement noise and
the geometry–measurement mismatch cannot be separated (bracketed only by the literature above);
magnitude only, no listening test; headline numbers come from cross-validation and one
validation split — the 36 locked subjects are still unspent (§9C 3a).

### 2026-10-07 — FFM26 held-out test, evaluated once

`notebooks/FFM_2_6_2QuadDB.ipynb`, evaluation path only (training cell skipped; `FFM26_best.pth`,
epoch 100, loaded and not rewritten). The scalar preflight first replayed validation exactly
(live 1.4855 dB = checkpoint 1.4855 dB), then read the test partition of `split_v21.json`.
Baseline: each database's own training-set mean HRTF, same band and pooling as `evaluate()`.

| Test | FFM26 RMSE (dB) | Population mean (dB) | Validation RMSE (dB) |
|---|---:|---:|---:|
| Pooled | **1.4606** (MAE 0.9158) | **4.8089** | 1.4855 |
| CAR Lab | 1.2521 | 4.772 | 1.3483 |
| HUTUBS | 1.7911 | 4.647 | 1.8345 |
| SONICOM | 1.4325 | 4.835 | 1.4545 |
| SYMARE | 1.2527 | 5.007 | 1.3261 |

Test is close to validation on every database (0.02–0.10 dB lower), so the validation figures
were not selected optimistically. The CAR Lab test baseline (4.772) equals FFM23's, as it should:
same split, same subjects. **The FFM26 test set is now spent**; no further model change may be
scored on it. Heatmaps (Cell 9) not run.

### 2026-10-10 — 48 kHz data pipeline; four faults in the delivered and exported data

**Why.** All databases move to one sampling rate, 48 kHz, so that bins line up. SYMARE and CAR Lab
are native 48 kHz and are no longer resampled. SONICOM ships 48 kHz measurements and, on its file
server, 48 kHz Mesh2HRTF simulations (`HRIR_SONICOM_48000.sofa`; 200 files fetched with rclone and
checked: the same 200 ids as the 44.1 kHz set, 48 kHz, (793, 2, 320), grid identical to the measured
one). HUTUBS, the only database held at 44.1 kHz alone, is upsampled. The 44.1 kHz arrays and FFM26
are left as they are.

**Code** — `preprocess_48k/` (commit c6c0b59): `preprocess_48k.py` (all six datasets),
`verify_48k.py`, `pack_48k.py`, and a README. Conventions fixed before the run:

- Spectrum = the DTFT of the complete impulse response at k × 187.5 Hz, k = 0…128; nothing is
  truncated. The SONICOM simulations are 320 samples long, and cutting them to 256 would remove up to
  21% of the energy in some directions. For a 256-sample IR this equals `rfft(h, n=256)`.
- HUTUBS: `resample_poly(160, 147)`, Kaiser β 10, gain 44100/48000. SciPy's default β 5 left a
  +0.006 dB mean / 0.018 dB rms passband error (0.11 dB at the 99.9th percentile, near deep notches);
  with β 10 the worst subject is 0.0002 / 0.0016 / 0.008 dB up to 16 kHz, checked per subject against
  the DTFT of the original.
- dB floor 1e-6; STPS order 2, λ 1e-5 (`spherical_spline.py`, identical to the copy in
  `preprocess_multidataset_stps.py` to 1.5e-14); one matrix per distinct subject grid, keyed by a
  hash of the full grid.
- Per file: sampling rate, `SourcePosition` type and units, receiver order and `ListenerView` are
  asserted. Per subject: the left ear must be louder (and, where the ITD is kept, earlier) at azimuth
  +45, +90, +135° and the reverse at −45, −90, −135°, with |ILD| > 3 dB at ±90° and > 1 dB
  elsewhere. The 1 dB bound was set after a first scan at 3 dB flagged SONICOM P0220 and HUTUBS
  pp33 for a correct-sign but weak ILD at −135° (−2.0, −1.8 dB); both are normal at the other five
  azimuths.
- Analysis band: bins 0–85 (≤ 15.94 kHz). Above 16 kHz CAR Lab has its 16.3 kHz filter, the
  SYMARE BEM is valid only to ~16 kHz, and HUTUBS has resampling images at 22.05–24 kHz.

**Faults found** (all handled in the 48 kHz data; §5C #23–24):

1. **CAR Lab grid key.** `export_carlab_npy.py` and `export_carlab_44k1.py` cached the STPS matrix
   under (number of directions, first direction). CL126, CL154 and CL159 each have 392 directions
   but different grids — each misses a different one of the 393 — so CL154 and CL159 were
   interpolated with CL126's grid: rms 2.0 dB (max 24 dB) and 0.2 dB (max 14 dB). CAR Lab has six
   distinct grids in all (196, 18, 1, 1, 1, 1 subjects).
2. **Ear channels exchanged** in the delivered data: CAR Lab CL003 (its `carlabdb.mat` remark reads
   "left and right mics reversed"), CL088 (no remark) and SONICOM P0339 (both its 44.1 and 48 kHz
   files). ILD and ITD are reversed at all six azimuths, front/back normal. Swapped back and flagged.
3. **SONICOM P0209** has inconsistent left/right cues rather than swapped ones: ILD reversed but
   weak (−11 / +7 dB at ±90°), ITD +396, −375, +1458 µs at +90, +135, −45°. Excluded. It has no
   mesh, so mesh → latent is unaffected.
4. **CAR Lab CL169** records six directions twice (azimuth +180° and −180° at six elevations). The
   copies are merged by averaging their dB magnitudes. They differ by 3.5 / 4.0 dB rms (left /
   right, 0.2–16 kHz) — one subject and rear directions only, but a direct look at
   repeat-measurement variability (cf. §9C 4c).

In FFM26's split, CL154 is in validation; CL159, CL003, CL088, P0339 and P0209 are in training;
none is in test. The reported FFM26 figures are not recomputed; the CAR Lab validation figure
(1.3483 dB) includes CL154's wrong-grid data.

**Checks of the output** (`verify_48k.py`, all passed): every array's SHA-256 against its manifest,
shape, dtype and finiteness, no unlisted files; every SONICOM simulation paired with a measurement,
SYMARE/BEM 61 = 61; the same code run at 44.1 kHz reproduces the existing 44.1 kHz arrays (SONICOM
P0001 max |diff| 3.8e-6 dB, HUTUBS pp1 0); 213 of 218 CAR Lab subjects identical to the old 48 kHz
export (< 1e-3 dB), the other five being exactly the intended changes; mirror permutation exact and
equal to the stored one.

The SONICOM 44.1 kHz simulations read by `physics_ceiling.py` (294 samples) were truncated to
256 there. Measured on 20 subjects, the effect is 0.07 dB rms (worst subject 0.14 dB), against a
4.7 dB simulation–measurement difference: no 10-02 conclusion changes.

**Output** (outside the repository): `ELEC5021/data_48k/` — SONICOM 398, SONICOM_SIM 200, HUTUBS 96,
SYMARE 61, SYMARE_BEM 61 (ids paired with SYMARE), CAR Lab 218 subjects; (129, 3072) float32 per
ear, right ears unmirrored; `meta.json` (records commit c6c0b59, clean), `manifest.json`,
`README_DATA.md`, `verification_report.json`. Packed for Drive as `data_48k_2026-10-10.zip`
(2.80 GiB, 2068 arrays, SHA-256 `f51c6b072123e70d6d8a0691373ab90e3f6db9856ce6306f65d949c87fbf97c7`).

**Decision.** A new autoencoder is trained on the 48 kHz data under its own name, separate from
FFM26 (86 bins). It is a different model with its own evaluation; FFM26's figures stay as reported.

### 2026-10-10 (later) — FFM48k_meas trained; the simulation check fails on SYMARE BEM

`notebooks/FFM48k_Train.ipynb` (commit 917dfcb) on Colab with `INCLUDE_SIM = False`: the four measured
databases at 48 kHz, 86 bins (DC–15.94 kHz), FFM26's architecture, hyperparameters and schedule,
100 epochs. The checkpoint records the hashes of the data zip and the split (f51c6b07…, dc89bc45…).
Results and figures: `results/FFM48k_meas_curves.png`, `results/FFM48k_meas_film.png`,
`results/FFM48k_meas_sim_check.json`.

| Validation | Subjects (train / val / test) | FFM48k_meas (dB) | Train-mean baseline (dB) | Reduction |
|---|---:|---:|---:|---:|
| Pooled | 567 / 85 / 121 | **1.4793** (MAE 0.9278) | — | — |
| CAR Lab | 174 / 22 / 22 | 1.3581 | 5.053 | 73.1% |
| HUTUBS | 58 / 14 / 24 | 1.8178 | 5.163 | 64.8% |
| SONICOM | 298 / 40 / 60 | 1.4445 | 4.983 | 71.0% |
| SYMARE | 37 / 9 / 15 | 1.3209 | 4.813 | 72.6% |

- The best epoch is 99 of 100 (epoch 100: 1.4794 dB); validation is within 0.01 dB of the best from
  epoch 76. Training error 1.3102 dB at the best epoch, a gap of +0.17 dB (FFM26: +0.15 dB).
- The side direction carries 1.0–3.8% of the latent variance (CAR Lab 1.4, HUTUBS 3.8, SONICOM 1.0,
  SYMARE 1.4%): mirroring is correct.
- FiLM is EQ-like (the rms second difference of β is 0.14–0.35 dB). In-band β is +6.3 dB (CAR Lab),
  +3.6 (HUTUBS), −11.3 (SONICOM) and +1.4 (SYMARE); |γ| ≤ 4.6 dB, largest at DC. The SONICOM offset is
  its free-field-compensated level.
- Not comparable with FFM26's 1.4855 dB: frequency axis, band and data differ. The baselines are close to
  FFM26's (5.053 / 5.163 / 4.983 / 4.813 against 5.053 / 5.162 / 4.977 / 4.805 dB), because the
  population spread over ~0–16 kHz depends little on the grid.

**Simulation check** (Cell 7b; the rule was fixed before training: on the same development subjects, a
simulation's reconstruction error must be ≤ 1.25 × that of the measurements, using the better of two
encodings):

| | Subjects | Measured (dB) | Simulated, partner FiLM | Simulated, no FiLM | Ratio | |
|---|---:|---:|---:|---:|---:|---|
| SONICOM_SIM | 164 | 1.379 | 3.348 | 1.284 | 0.93 | pass |
| SYMARE_BEM | 46 | 1.136 | 1.593 | 1.668 | 1.40 | fail |

The rule requires both, so it fails, and `FFM48k_measim` (`INCLUDE_SIM = True`: the two simulated sets
added with their own FiLM rows, each subject in its measured partition) is trained for Task 1. The
thesis's autoencoder results stay on `FFM48k_meas`. A per-database decision (SONICOM through
`FFM48k_meas`, SYMARE through `FFM48k_measim`) was not taken: the rule was set for both. SONICOM's
partner FiLM fails on the simulations because it carries the −11 dB measured level; without FiLM the
simulations reconstruct better than the measurements.

The test partition has not been read (Cell 8, single use, at the end).

### 2026-10-11 — FFM48k_measim; Task 1: geometry predicts the simulations better than the measurements

**FFM48k_measim.** `notebooks/FFM48k_Train.ipynb` with `INCLUDE_SIM = True`: the four measured
databases plus SONICOM_SIM and SYMARE_BEM, each simulated subject in its measured partner's partition
(SONICOM_SIM 147 / 17 / 36, SYMARE_BEM 37 / 9 / 15), own FiLM rows 4 and 5; same split, architecture,
hyperparameters and schedule as FFM48k_meas. Best epoch 100 of 100.

| Validation | FFM48k_measim (dB) | FFM48k_meas (dB) |
|---|---:|---:|
| Pooled, six sets | 1.3883 | — |
| Pooled, four measured | 1.4640 | 1.4793 |
| CAR Lab | 1.3423 | 1.3581 |
| HUTUBS | 1.8044 | 1.8178 |
| SONICOM | 1.4237 | 1.4445 |
| SYMARE | 1.3285 | 1.3209 |
| SONICOM_SIM | 1.0109 | — |
| SYMARE_BEM | 1.2640 | — |

- Adding the simulations does not cost the measured databases anything (pooled −0.015 dB; SYMARE
  +0.008 dB).
- The simulations are now in distribution: 0.71 × (SONICOM_SIM) and 0.95 × (SYMARE_BEM) their measured
  partners' validation error.
- In-band FiLM β: SONICOM −13.9 dB, SONICOM_SIM +4.1 dB, SYMARE_BEM +6.4 dB — the measured/simulated
  level difference is absorbed by FiLM, not z.
- The thesis's autoencoder results stay on FFM48k_meas; FFM48k_measim is the encoder for Task 1 only.
  Its test partition has not been read.

**Task 1** (`task1/Task1_SimVsMeas.ipynb`, Colab, run 10-11; results `results/task1_results.json`,
`results/task1_r2_per_bin.png`). The question is the workplan's: is geometry → HRTF weak because the
relation cannot be learned from our features and N (cause A), or because the measured HRTF does not
follow the scanned geometry (cause B)? A simulation is a deterministic function of the mesh with no
measurement in it, so the same features, regressor, subjects and folds are used to predict the simulated
and the measured HRTF. Design fixed in the notebook header before the run:

- Subjects: SONICOM 164 development subjects, folds identical to FFM_3_2_1 (asserted); SYMARE 46
  (AE train + val). Locked test subjects (36 / 15) not read. One row per ear, right ear mirrored.
- Geometry: SONICOM FFM_3_2_1's `simple+ear multi-σ` (its best configuration on FFM26 measured z, so any
  selection bias favours the measured target); SYMARE the 19 ipsilateral landmarks of FFM_2_8.
- Targets: z_meas, z_sim (FFM48k_measim, each domain through its own FiLM row), and DTF_meas, DTF_sim
  without the autoencoder (dB, bins 1–85, the 2652 directions at elevation ≥ −45°, each ear's
  direction-mean spectrum removed).
- Regressor: FFM_3_2_1's kernel ridge rewritten in dual form (target Gram matrices, any target
  dimension), α by inner 5-fold grouped CV over 37 values. Gate: the same engine and features on FFM26
  z must reproduce FFM_3_2_1 — **+0.0757 vs +0.0757**. No fold chose an α on the grid edge.
- Controls: geometry permuted by subject (100); calibration (the subject's own simulation as the input);
  geometry + simulation (unit-trace kernels summed); paired subject bootstrap (1000).

CV-R² (z / DTF):

| | SONICOM (164) | SYMARE (46) |
|---|---|---|
| Geometry → measured | 0.069 / 0.069 | 0.025 / 0.051 |
| Geometry → simulated | **0.197 / 0.173** | 0.074 / 0.065 |
| Simulated − measured, 95% CI | z +0.127 [+0.111, +0.142]; DTF +0.103 [+0.090, +0.117] | z +0.049 [+0.033, +0.065]; DTF +0.014 [+0.002, +0.026] |
| Own simulation → measured | 0.091 / 0.091 | 0.080 / 0.142 |
| Geometry + simulation → measured | 0.097 / 0.099 | 0.056 / 0.119 |
| (Geometry + simulation) − geometry, 95% CI | z +0.028 [+0.022, +0.034]; DTF +0.029 [+0.022, +0.036] | z +0.030 [+0.022, +0.038]; DTF +0.068 [+0.055, +0.080] |

By band (geometry → simulated vs → measured, z / DTF):

| | 0.2–5 kHz | 6–12 kHz | 12–16 kHz |
|---|---|---|---|
| SONICOM simulated | 0.50 / 0.51 | 0.23 / 0.20 | 0.13 / 0.11 |
| SONICOM measured | 0.12 / 0.10 | 0.07 / 0.07 | 0.05 / 0.04 |
| SYMARE simulated | 0.28 / 0.26 | 0.07 / 0.08 | 0.04 / 0.03 |
| SYMARE measured | 0.03 / 0.05 | 0.03 / 0.06 | 0.01 / 0.04 |

End to end (z targets decoded; pooled RMSE over subject × ear × direction × bin; gap closed between the
training-fold mean and the AE reconstruction):

| | SONICOM | SYMARE |
|---|---|---|
| Geometry → measured | 4.641 dB (mean 4.815, AE 1.339): **5.0%** | 4.710 dB (mean 4.743, AE 1.080): **0.9%** |
| Geometry → simulated | 3.768 dB (mean 4.244, AE 0.891): **14.2%** | 4.128 dB (mean 4.282, AE 1.158): **4.9%** |
| Own simulation → measured | 4.587 dB: 6.6% | 4.554 dB: 5.2% |
| Geometry + simulation → measured | 4.563 dB: 7.3% | 4.633 dB: 3.0% |
| Gap closed, simulated − measured, 95% CI | +9.2 points [+7.8, +10.6] | +4.0 points [+2.6, +5.4] |

All four geometry runs in both databases beat all 100 permutations (p = 0.0099, the floor for 100;
null 95th percentiles ≤ +0.0016).

Reading:

- **Both causes act, in different bands.** Below 5 kHz the features explain half of the simulated
  variance on SONICOM (a quarter on SYMARE) but a tenth or less of the measured: the measured HRTF does
  not follow the scanned geometry there (B). Above 6 kHz, where the pinna acts, even the simulations are
  poorly predicted — SONICOM ≤ 0.23, SYMARE ≤ 0.08 — so with these features and N the pinna relation is
  not learned even without measurement in the target (A). The measured values are lower again.
- **The calibration bounds what a better geometry → simulation model could give.** The subject's own
  simulation explains R² 0.09 of the measured HRTF on SONICOM (0.08 / 0.14 on SYMARE), 6.6% / 5.2% of the
  gap. It does carry information the features miss (geometry + simulation beats geometry, CIs exclude 0),
  but not much. Consistent with the per-side BEM–measurement correlation of 0.16–0.23 (§2C, 10-02).
- **Neither simulation includes the torso.** SYMARE `HE16kHz` is the "Head and ears" mesh, no torso
  (SYMARE readme; the `HTE05kHz` set, head, torso and ears, valid to about 5 kHz, is held locally for all
  61 subjects). The SONICOM meshes end at the neck: lowest point 81–148 mm (median 111) below the
  interaural axis, the lowest 2 cm 90–141 mm wide. Torso and shoulder effects in the measurements are
  therefore a candidate for the low-band gap — in neither the scan features nor the simulations.
  *Tested the same day (Task 1b, below): not supported.*
- On SYMARE, geometry + simulation is below simulation alone (z 0.056 vs 0.080, DTF 0.119 vs 0.142): the
  two kernels enter with equal weight and one α, and the 19-landmark kernel dilutes the simulation one.

Caveats:

- The comparison is conservative for the simulations (the feature set was chosen on measured z).
- R² is relative to each target's own spread, and the simulations spread less (SONICOM 4.24 vs 4.82 dB
  about the mean; SYMARE 4.28 vs 4.74). The dB gap-closed comparison, normalised the same way for both
  domains, agrees.
- The AE reconstruction errors here (1.339 / 0.891 / 1.080 / 1.158 dB) include AE training subjects and
  are optimistic; this only enlarges the gap-closed denominators, equally for both domains. The DTF
  targets bypass the AE and give the same reading (SONICOM measured 0.069 by both routes).
- The SYMARE DTF difference is marginal (CI [+0.002, +0.026]; N = 46, 19 landmarks); the SONICOM result
  carries the conclusion.
- Low-energy bins can show large per-bin R² (SYMARE DTF_meas 0.40 at 187.5 Hz; SONICOM z_sim −0.05 at DC);
  the band figures weight bins by variance.

### 2026-10-11 (later) — Task 1b: adding the torso to the simulation does not close the low-band gap

The torso candidate above, tested on SYMARE, which ships the same FM-BEM on two meshes: `HE16kHz` (head
and ears, used in Task 1) and `HTE05kHz` (head, torso and ears, valid to about 5 kHz; the BEM-ready meshes
keep the longest edge below λ/6 at their maximum frequency, so both are adequate below 5 kHz).

**Data.** `preprocess_48k.py` gained `SYMARE_HTE` (commit 3863f29), written on request only, to its own
folder; the frozen `data_48k_2026-10-10.zip` is unchanged. Below 5.06 kHz HTE and HE agree to 2.1–3.7 dB
rms per subject; above 5.3 kHz HTE rolls off (−13 dB at 5.8 kHz, ≤ −19 dB from 6.2 kHz). Its left/right
check therefore uses the ILD over 1.5–5 kHz (over 1.5–16 kHz the rolled-off spectrum gives the wrong sign
at ±135° for some subjects); all 61 pass, ITD at ±90° 646–833 µs. `pack_hte.py` checks it against
`SYMARE_BEM`: same ids, grid and pairing, distinct sources, every array hash. Pairing has to be checked on
deviations from each set's population mean: on raw spectra only 5 / 61 subjects' HTE is nearest their own
HE, because the torso shifts everyone by about the same 2–3 dB, more than subjects differ; on deviations
56 / 61 (left) and 50 / 61 (right), correlation 0.59 with the own subject against 0.00 with others.
`data_48k_hte_2026-10-11.zip`, SHA-256 0b0f9ab8….

**Design** (`task1/Task1b_TorsoCheck.ipynb`, fixed in its header before the run; Colab, CPU): the 46 SYMARE
development subjects and Task 1's folds; 0.2–5 kHz (bins 2–26), measured DTF as in Task 1 (no
autoencoder). Primary: Task 1's calibration — the subject's own simulated DTF → measured DTF by kernel
ridge — run with HTE and with HE; rule: the torso explanation is supported if the paired-bootstrap 95% CI of
R²(HTE → meas) − R²(HE → meas) lies above 0. Secondary (descriptive): per-ear correlation ρ of simulated and
measured deviations from the same-side mean; geometry (Task 1's 19 landmarks: pinna, midsagittal and one eye;
none on the torso, §5C #25) → HE, HTE and measured.

*Disclosure.* While diagnosing the pairing check, before the notebook was written, the assistant printed
descriptive correlations between simulated and measured deviations on all 61 SYMARE subjects, the 15
locked ones included (raw dB, 0.2–5 kHz: HE about 0.33, HTE about 0.28). Nothing was fitted; the design is
Task 1's calibration and was not changed after.

**Results** (46 subjects, 0.2–5 kHz):

| | HE (head + ears) | HTE (head + torso + ears) | HTE − HE, 95% CI |
|---|---:|---:|---|
| Own simulation → measured, CV-R² (primary) | 0.104 | 0.087 | **−0.017 [−0.036, −0.002]** |
| ρ with the measured deviations (mean over ears) | 0.279 | 0.210 | −0.069 [−0.096, −0.041] |
| Geometry → this simulation, CV-R² | 0.370 | 0.128 | −0.241 [−0.281, −0.197] |
| Geometry → measured, CV-R² | 0.050 | | |

Both calibrations beat all 100 subject permutations (p = 0.0099); no fold chose an α on the grid edge.
HTE − HE differ by 2.0 dB rms in this DTF. By sub-band (mean per-bin R², own simulation → measured):
0.4–1.5 kHz HE 0.054 vs HTE 0.065; 1.5–3 kHz 0.072 vs 0.063; 3–4.9 kHz 0.130 vs 0.103. The values differ
from Task 1's 0.2–5 kHz band parts because the regression is fitted on this band alone.

Reading:

- **By the rule fixed beforehand, there is no evidence that the missing torso explains the low-band gap.**
  The simulation with the torso predicts the measured HRTF slightly *worse*, the CI excluding 0 on the
  negative side, and its deviations correlate less with the measured ones. Where shoulder reflections act
  (about 1–3 kHz) HTE does not help either.
- The torso does add individual variation to the simulation (2.0 dB rms), most of it not predictable from
  head and ear landmarks (geometry → simulation 0.37 → 0.13). That is the same signature the measurements
  show (geometry → measured 0.05). But the torso as scanned does not reproduce the measured individual
  pattern. The low-band finding of Task 1 (cause B) stands; "the simulations lack a torso" is withdrawn as
  its explanation. Remaining candidates, untested: a different posture in the scan and in the measurement
  (head–torso angle, shoulders), and the measurement side (low-frequency equalisation and compensation,
  positioning).
- Caveat: the HTE meshes are coarser (BEM-ready for 5 kHz rather than 16 kHz). Both meet the λ/6 criterion
  below 5 kHz, but a mesh-resolution contribution to the HTE − HE difference cannot be excluded.

### 2026-10-12 — Task 2, first version: shape–HRTF modes on named shape measures

`task2/Task2_ShapeModes.ipynb` (commit 2ed3553; Colab, run 10-11; results `results/task2_results.json`,
`results/task2_modes_sonicom.png`, `results/task2_modes_symare.png`). Exploratory, shallow first: only shape
inputs that are already in correspondence and can be read directly. SONICOM (164): FFM_3_2_1's nine simple
measures, head (half inter-canal distance, head length, head height) and ear (canal fore–aft and up–down,
pinna protrusion, height and length, ear-crop area), each standardised. SYMARE (46): landmark coordinates in
the common head frame, in mm, head (5 midsagittal points and both eyes) and ear (13 ipsilateral pinna
points). Targets: Task 1's simulated and measured DTFs (no autoencoder). Task 1's subjects and folds. Gate:
Task 1's geometry → measured DTF reproduced exactly (0.0693, 0.0509).

Design fixed in the header: (1) ridge prediction from head, ear and all inputs, with reduced-rank versions,
permutations and subject bootstrap; (2) one shape → DTF ridge map per domain, **the same α for both**
(the median fold choice for all → simulated: relative α 10 in both databases), modes from the SVD of the
simulated map's fitted values; mode *k*'s **shape change** is the population-average change in shape per
+1 SD of its score, and both maps' **responses to that same shape change** are compared (cosine: same
pattern?; rms ratio measured / simulated: how much?), overall and by band; 1000 permutations (measured
shuffled by subject) and 1000 bootstraps; (3) reading: a mode survives in measurement if its cosine has
permutation p < 0.05 and a CI above 0.

**Prediction** (CV-R², DTF):

| | SONICOM sim | SONICOM meas | SYMARE sim | SYMARE meas |
|---|---:|---:|---:|---:|
| Head | 0.021 | 0.016 | 0.019 | 0.013 |
| Ear | 0.070 | 0.034 | 0.059 | 0.045 |
| All | 0.072 | 0.037 | 0.048 | 0.034 |
| All, 0.2–5 / 6–12 / 12–16 kHz | 0.32 / 0.08 / 0.04 | 0.07 / 0.04 / 0.02 | 0.17 / 0.06 / 0.02 | 0.03 / 0.04 / 0.02 |
| All, rank 1 / 2 / 3 | 0.040 / 0.065 / 0.071 | 0.022 / 0.038 / 0.037 | 0.044 / 0.045 / 0.040 | 0.026 / 0.030 / 0.030 |

- All four "all" fits beat 100 permutations (p = 0.0099). The ear measures carry most of it: ear over head
  +0.051 [+0.040, +0.062] (SONICOM sim), +0.022 [+0.015, +0.028] (meas); SYMARE +0.029 [+0.019, +0.039],
  +0.020 [+0.013, +0.027]. Simulated − measured: +0.035 [+0.025, +0.044] (SONICOM), +0.014 [+0.002, +0.026].
- The nine named measures recover about 40–55% of what Task 1's richer features did (0.072 vs 0.173
  simulated, 0.037 vs 0.069 measured). Three modes hold almost all of it on SONICOM (rank 3: 0.071 of
  0.072). On SYMARE, 60 inputs for 46 subjects: all is below ear alone.

**Modes** (simulated map's modes; responses of both maps to the same shape change):

| | Share of simulated fit | Shape change at +1 SD | rms response sim / meas (dB) | meas / sim, 95% CI | Cosine, 95% CI; perm. p | Cosine 0.2–5 / 6–12 / 12–16 kHz | Stability (median, 5%) |
|---|---:|---|---|---|---|---|---|
| SONICOM 1 | 46% | larger overall: ear-crop area +567 mm², head length +5.8 mm, head height +3.2, pinna height +1.6, inter-canal +1.4 ×2, protrusion +1.4 | 0.68 / 0.59 | 0.86 [0.79, 0.97] | +0.31 [+0.20, +0.39]; 0.012 | **0.83** / 0.29 / 0.03 | 0.98, 0.89 |
| SONICOM 2 | 26% | protruding pinna: protrusion +2.2 mm, with a shorter head (−4.4 mm) and a smaller pinna (length −2.2, height −1.5) | 0.51 / 0.45 | 0.88 [0.81, 1.01] | **+0.73** [+0.56, +0.73]; 0.001 | 0.87 / **0.72 / 0.66** | 0.93, 0.78 |
| SONICOM 3 | 9% | longer pinna (+6.7 mm), taller head (+2.5), smaller crop area | 0.30 / 0.24 | 0.79 [0.77, 1.03] | +0.39 [+0.16, +0.46]; 0.001 | 0.55 / 0.45 / 0.33 | 0.89, 0.45 |
| SYMARE 1 | 65% | larger head: taller (vertex +5.0, chin −4.0 mm), ears 3.5 mm more lateral | 0.69 / 0.62 | 0.90 [0.85, 1.00] | +0.54 [+0.34, +0.60]; 0.002 | 0.52 / 0.59 / 0.39 | 0.95, 0.83 |
| SYMARE 2 | 13% | head shape: occiput +10.3 mm higher, vertex +5.4 mm forward, ears 2 mm forward | 0.30 / 0.33 | 1.08 [0.93, 1.23] | +0.52 [+0.27, +0.56]; 0.001 | 0.39 / 0.69 / 0.32 | 0.80, 0.18 |
| SYMARE 3 | 6% | face points forward (3–6 mm) | 0.22 / 0.24 | 1.11 [0.94, 1.24] | +0.27 [+0.13, +0.42]; 0.001 | 0.30 / 0.28 / 0.27 | 0.61, 0.08 |

The first simulated mode is stronger than under permutation in both databases (p = 0.001). Permutation
nulls of the cosine have 95th percentiles +0.15 to +0.36. A derived summary, the part of the simulated
response that reappears in the measured one along the same pattern (ratio × cosine): SONICOM mode 1 0.78
below 5 kHz, 0.23 at 6–12, 0.03 at 12–16 kHz; mode 2 0.74 / 0.63 / 0.60.

Reading (by the rule fixed beforehand, all six modes survive in measurement; read per band):

- **The shape-driven change is in the measurements at nearly full size.** Along the same shape change the
  measured response is 0.79–1.11 × the simulated one. The protruding-pinna mode (SONICOM 2) has the same
  pattern in measurement in every band, the pinna range included (cosine 0.72 and 0.66 at 6–12 and
  12–16 kHz). The overall-size mode (SONICOM 1) matches below 5 kHz (0.83) but not its pinna-band pattern
  (0.29, 0.03).
- **This explains Task 1's low band.** Between people, the measured DTF varies 3.1 × (SONICOM) and 3.5 ×
  (SYMARE) as much as the simulated one at 0.2–5 kHz (rms 1.72 vs 0.97 and 1.79 vs 0.96 dB; local check on
  the same subjects, 10-12), and about equally at 6–16 kHz (1.0–1.3 ×). Shape moves the measured low band as
  much as the simulated one, and in the same pattern, but the measured low band carries about three times
  more between-person variation that shape does not explain, so R² falls (0.32 → 0.07 here, 0.50 → 0.10 in
  Task 1). Cause B in the low band is therefore extra variance in the measurement, not a different response
  to shape; Task 1b showed the scanned torso does not account for it.
- **Above 6 kHz the limit is the simulated relation itself** (Task 1's cause A): the named measures explain
  ≤ 0.08 of the simulated DTF there, and of what they do explain, one pattern (protrusion) survives in
  measurement and another (overall size) largely does not.
- The measured map's own leading shape directions resemble the simulated ones (|cosine| 0.96 / 0.83 / 0.66
  on SONICOM). On SYMARE (1.00 / 0.99 / 0.96) this says little: with α = 10 and 60 inputs for 46 subjects, the
  fitted modes are dominated by the inputs' own leading variation whatever the target.

Caveats:

- Between-person associations, not one person's sensitivity to their own ear (that would need perturbed
  meshes and new BEM solves).
- Correlated measures are not separable: mode 2 couples a protruding pinna with a shorter head and smaller
  pinna; "protrusion" names the largest loading, not a cause.
- The bootstrap distribution of a cosine is pulled towards 0 by estimation noise, so several observed
  cosines sit at the top of their percentile CIs (SONICOM 2: 0.73 in [0.56, 0.73]). The ratio can be inflated
  by noise in the measured map; ratio × cosine is the conservative figure.
- SYMARE modes 2–3 are unstable (5th-percentile stability 0.18, 0.08), and SYMARE's landmark names are
  inferred from positions (the readme does not list them).
- Two databases × three modes, no correction for multiple comparisons; exploratory.

### 2026-10-12 (later) — Task 2 figures redrawn; the between-person variance made reproducible

- `task2/redraw_task2_figures.py` redraws the two mode figures at print size from the saved
  `task2_responses.npz`, without refitting; it first recomputes every statistic in
  `results/task2_results.json` from the npz (6 modes × 4 bands × rms, cosine, ratio; max difference 7e-10).
- The between-person spread table (simulated vs measured DTF, 0.2–5 / 6–12 / 12–16 kHz) had come from a
  local check with no script kept. `task2/between_person_variance.py` recomputes it from the 48 kHz data
  (zip SHA-256 as in the Task 2 notebook, manifests byte-identical to the zip, every array hash-checked;
  same 164 / 46 subjects and DTF) and reproduces every value; definition: rms over ears × band bins ×
  directions of the deviation from the across-ear mean (ddof 0). Subject-bootstrap 95% intervals of the
  variance ratio measured / simulated, added now:

| | 0.2–5 kHz | 6–12 kHz | 12–16 kHz |
|---|---|---|---|
| SONICOM (164) | 0.97 / 1.72 dB, **3.1 × [2.9, 3.4]** | 3.98 / 4.56, 1.3 × [1.3, 1.4] | 4.56 / 4.63, 1.0 × [1.0, 1.1] |
| SYMARE (46) | 0.96 / 1.79 dB, **3.5 × [3.0, 4.1]** | 4.23 / 4.26, 1.0 × [1.0, 1.1] | 4.66 / 4.48, 0.9 × [0.9, 1.0] |

  Output: `results/task2_between_person_variance.json`.

### 2026-10-12 (later) — Autoencoder LSD: FFM26 (validation and test) and the 48 kHz models (validation)

`notebooks/AE_LSD_Report.ipynb`, evaluation only (no training, no checkpoint written); output
`results/ae_lsd_report.json`. Until now only the mesh experiments reported LSD; the autoencoders were in pooled RMSE only.

- Metrics in one pass per ear: pooled RMSE (as `evaluate()`), LSD (per ear and direction RMS over the band's bins, then
  the arithmetic mean over directions and ears — the same definition as FFM_3_2_1 and FFM_2_6 Cell 9), and LSD over the
  2652 directions at elevation ≥ −45°. Baseline: each database's training-set mean HRTF on the same partition.
  Bins as in training: FFM26 93 (DC–15.85 kHz, 44.1 kHz); FFM48k 86 (DC–15.94 kHz, 48 kHz).
- Partitions: FFM26 validation and test — the test set was spent on 10-07; this adds a metric to the same frozen
  checkpoint, with no model change or selection. FFM48k_meas and FFM48k_measim: validation only; the notebook builds no
  test partition, so **the 48 kHz test set is still unspent**.
- Gate before any LSD was reported: every recorded RMSE (model and baseline, per database and pooled) reproduced within
  0.005 dB; largest model difference 0.00005 dB (FFM26 test CAR Lab 1.25205 vs 1.2521 recorded). The thesis keeps the
  recorded RMSE values. Data and split checked by SHA-256 (48 kHz) and against the checkpoint splits.

| | AE RMSE | Mean RMSE | AE LSD | Mean LSD | AE LSD el ≥ −45° | Mean LSD el ≥ −45° |
|---|---:|---:|---:|---:|---:|---:|
| FFM26 val, pooled (170 ears) | 1.4855 | 5.0097 | 1.3169 | 4.7672 | 1.3832 | 4.7078 |
| FFM26 test, pooled (242 ears) | 1.4606 | 4.8089 | **1.2990** | **4.6044** | 1.3576 | 4.5460 |
| FFM48k_meas val, measured (170) | 1.4793 | 5.0139 | 1.3101 | 4.7696 | 1.3772 | 4.7095 |
| FFM48k_measim val, measured (170) | 1.4640 | 5.0139 | 1.3042 | 4.7696 | 1.3692 | 4.7095 |
| FFM48k_measim val, SONICOM simulations (34) | 1.0109 | 4.4150 | 0.8454 | 4.2035 | 0.9113 | 4.2221 |
| FFM48k_measim val, SYMARE BEM (18) | 1.2640 | 4.1553 | 1.0923 | 4.0339 | 1.1685 | 4.0296 |
| FFM48k_measim val, all six (222) | 1.3883 | 4.8616 | 1.2167 | 4.6233 | 1.2828 | 4.5797 |

  Per database (FFM26 test, AE / mean LSD): CAR Lab 1.152 / 4.595, HUTUBS 1.633 / 4.477, SONICOM 1.256 / 4.628,
  SYMARE 1.153 / 4.727. Per-database values for every run are in the JSON.
- Reading: LSD is below pooled RMSE throughout (a mean of per-direction RMS values against a root of the overall mean),
  and the reduction against the mean is about the same in both metrics (FFM26 test 69.6% RMSE, 71.8% LSD). Over
  el ≥ −45° the autoencoder's LSD is higher than over the full sphere for every set except HUTUBS, and the mean's is
  lower for every set except the SONICOM simulations; the thesis tables use the full sphere, as Table mesh-ridge does
  (mean 4.646 dB).
- Thesis: LSD columns added to `tab:ae` and `tab:ae48k` (RMSE columns unchanged); pooled and simulated baselines
  filled from the JSON; Appendix A row added.

### 2026-10-12 (later) — mesh → latent redone at 48 kHz and scored as S2RNF scores (development set); test registered

`mesh2z/FFM_3_8_Mesh2Z_48k.ipynb` (Colab; outputs `results/mesh2z_48k_v1/`). This reopens §9C 2b for one purpose: a
comparison with Masuyama et al. (Interspeech 2026, S2RNF), which uses the same 200 SONICOM subjects. Their "RMSE" is LSD
(their §4.1 and Eq. 3), computed to 20 kHz without DC on the 793 measured directions: S2RNF 4.90 dB against the average
HRTF's 5.05 (3.0% lower) and MAE 3.60 against 3.78 (4.8% lower), on 25 test subjects. The two splits cross (their
`config.yaml`): of their 25 test subjects, 21 are in our AE training set, 2 in validation and 2 in test; of our 36 locked
subjects, 31 are in their training set, 3 in validation and 2 in test. Only each method against its own mean can be compared.

- **Design.**
  - Encoder: `FFM48k_measim`, frozen.
  - Subjects and folds: the 164 development subjects with FFM_3_2_1's folds.
  - Features: FFM_3_2_1's `simple + ear multi-σ` RFF, not reselected.
  - Regressor: kernel ridge, with α chosen by inner CV on the latent error.
  - **New:** standardisation and kernel scaling are estimated inside each training fold. FFM_3_2_1 and Task 1 estimated
    them on all 164 subjects (inputs only).
  - Rows (→ measured z): geometry; simulation (the subject's own z_sim); geometry + simulation.
  - References: the training mean; the raw simulation; the simulated deviation plus the measured mean; the AE
    reconstruction. The last uses the measured input, so it is a reference, not a bound.
  - Metrics: LSD and MAE on bins 1–85 (187.5 Hz–15.94 kHz, DC excluded) at elevation ≥ −45° (primary); the same LSD
    over the full sphere; the old pooled RMSE.
- **Gate.** With global standardisation the notebook reproduces Task 1 exactly. Fold-internal standardisation changes
  R²(z) by ≤ 0.0002 and pooled RMSE by ≤ 0.0007 dB.

| Development (164, out-of-fold) | R²(z) | LSD ≥ −45° | vs mean | MAE ≥ −45° | vs mean | LSD, full sphere | pooled RMSE (old) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training mean | — | 4.558 | — | 3.502 | — | 4.606 | 4.815 |
| Geometry | 0.069 | 4.378 | −4.0% | 3.329 | −4.9% | 4.429 | 4.641 |
| Simulation | 0.091 | 4.316 | −5.3% | 3.285 | −6.2% | 4.370 | 4.586 |
| Geometry + simulation | 0.097 | **4.292** | **−5.8%** | **3.255** | **−7.0%** | 4.346 | 4.563 |
| Simulated deviation + measured mean | — | 5.565 | +22.1% | 4.086 | +16.7% | 5.595 | 5.828 |
| Raw simulation | — | 19.14 | | 18.23 | | 19.76 | 20.15 |
| AE reconstruction (reference) | 1 | 1.288 | | 0.911 | | 1.186 | 1.339 |

- **Paired comparisons** (per subject, LSD ≥ −45°, 2000-resample bootstrap):
  - Against the mean: geometry −0.181 dB [−0.213, −0.149]; simulation −0.242 [−0.278, −0.208]; geometry + simulation
    −0.267 [−0.304, −0.229].
  - Geometry + simulation against geometry alone: −0.086 [−0.105, −0.065].
  - All one-sided paired t-test p values are < 1e-12.
  - Permutation test (geometry → z, 100 permutations): p = 0.010, the minimum possible.
  - α is the same in every fold (grid index 19, or 18 for the combined row), away from the edges.
- **By band** (RMS over directions, reduction against the mean):

  | Band | Geometry | Simulation | Both |
  |---|---:|---:|---:|
  | 0.2–5 kHz | 7.2% | 6.9% | 8.9% |
  | 5–10 kHz | 5.6% | 8.6% | 8.9% |
  | 10–16 kHz | 2.5% | 2.9% | 3.4% |

- **Against the 44.1 kHz result.** FFM_3_2_1 went from 4.797 to 4.631 dB pooled against the same 164-subject mean (3.5%);
  its published 4.857 baseline came from a 339-subject pool (§2C 10-02). Here the figures are 4.815 → 4.641 (3.6%). The
  move to 48 kHz and the data fixes leave the geometry result unchanged.
- **Raw simulation, 19.1 dB, cannot be compared with S2RNF's Mesh2HRTF row (10.53).** Their simulations are the
  `*_Windowed_NoITD_Scaled.sofa` files; ours (`HRIR_SONICOM_48000.sofa`) carry an overall level offset. Removing the
  training-mean difference brings ours to 5.57, which is still worse than the mean (consistent with §2C 10-02).
- **Reading.** On the development set the three rows are 4.0–5.8% below the mean in LSD (S2RNF on its test set: 3.0%)
  and 4.9–7.0% below it in MAE (S2RNF: 4.8%). This is not like for like:
  - development cross-validation against a held-out test set;
  - 16 kHz against 20 kHz;
  - HEALPix (interpolated) against the native grid;
  - different SOFA versions;
  - the AE was trained on these 164 subjects' HRTFs, although the regressor never saw a subject in its own fold.

**Test registration** (written before the 36 locked subjects are read; this settles §9C 3a). Cell 9 of FFM_3_8 is run once.
- Rows as above. **Primary: geometry, LSD on bins 1–85 at elevation ≥ −45°.**
- α is chosen by 5-fold CV on all 164 development subjects and the model refitted on all 164; the mean also comes from these 164.
- Reported:
  - paired bootstrap CIs and one-sided t-tests of each row against the mean, and of the combined row against geometry;
  - a calibration of the mean and the raw simulation on S2RNF's 160 / 25 split, under our protocol (no model).
- Expected precision: the development paired SD is 0.21 dB, which gives a 95% half-width of about 0.07 dB at N = 36.
- The marker is shared with FFM_3_5, 3_6 and 3_7. The cell also reads `FFM48k_measim` on 36 of its 60 SONICOM test subjects.

### 2026-10-12 (later) — FFM_3_8 on the 36 locked subjects, evaluated once

Cell 9 of `mesh2z/FFM_3_8_Mesh2Z_48k.ipynb` was run as registered above, and nothing was changed after the development
results. Output: `results/mesh2z_48k_v1/test_results.json`. The 36 locked SONICOM subjects are now spent, and the shared
marker blocks FFM_3_5, 3_6 and 3_7; §9C 3 and 3a are done. Training used the 164 development subjects, with α at grid
index 19 / 19 / 18 as on the development set.

| Test (36 subjects) | R²(z) | LSD ≥ −45° | vs mean | MAE ≥ −45° | vs mean | LSD, full sphere | pooled RMSE (old) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training mean (164) | — | 4.387 | — | 3.369 | — | 4.415 | 4.594 |
| **Geometry (primary)** | 0.055 | **4.252** | **−3.1%** | 3.244 | −3.7% | 4.287 | 4.469 |
| Simulation | 0.101 | 4.106 | −6.4% | 3.105 | −7.8% | 4.143 | 4.329 |
| Geometry + simulation | 0.099 | 4.107 | −6.4% | 3.108 | −7.7% | 4.148 | 4.332 |
| Simulated deviation + measured mean | — | 5.590 | +27.4% | 4.103 | +21.8% | 5.604 | 5.828 |
| Raw simulation | — | 18.96 | | 18.00 | | 19.59 | 19.97 |
| AE reconstruction (reference) | 1 | 1.326 | | 0.935 | | 1.221 | 1.380 |

- **Paired comparisons** (per subject, LSD ≥ −45°, 2000-resample bootstrap, one-sided paired t-test):
  - **Primary:** geometry against the mean, −0.134 dB [−0.204, −0.068], p = 0.00025.
  - Simulation against the mean: −0.280 [−0.378, −0.179], p = 1e-6.
  - Geometry + simulation against the mean: −0.279 [−0.374, −0.188], p = 8e-7.
  - Geometry + simulation against geometry alone: −0.145 [−0.219, −0.074], p = 0.0002.
  - MAE, full-sphere LSD and the old pooled RMSE give the same signs and similar sizes.
- **Development against test.**
  - Geometry: −4.0% → −3.1% (0.181 → 0.134 dB), within the precision expected at N = 36 (about ±0.07 dB).
  - Simulation: −5.3% → −6.4%. Geometry + simulation: −5.8% → −6.4%.
  - On the test set, adding geometry to the simulation gains nothing (4.107 against 4.106). This comparison was not
    registered: geometry + simulation is not better than simulation alone.
  - Pooled-RMSE gap closed (mean → AE reconstruction): geometry 3.9%, simulation 8.2%, both 8.1%. Development: 5.0 / 6.6 / 7.3%.
- **Calibration on S2RNF's split, under our protocol** (no model; mean from their 160 training subjects, scored on their
  25 test subjects):

  | | Ours | Theirs |
  |---|---:|---:|
  | Mean, LSD ≥ −45° | 4.732 | 5.05 |
  | Mean, LSD full sphere | 4.777 | |
  | Mean, MAE ≥ −45° | 3.679 | 3.78 |
  | Raw simulation, LSD ≥ −45° | 19.46 | 10.53 |

  - Our protocol puts the mean about 0.3 dB lower in LSD, as expected without 16–20 kHz.
  - Their 25 test subjects are harder than our 36 under the same protocol (mean 4.73 against 4.39). Absolute dB values
    from the two test sets are therefore not comparable; only reductions against each set's own mean are.
  - The raw-simulation gap comes from the SOFA versions (their `Scaled` files against ours).
- **Against S2RNF** (reductions against each method's own training mean, each on its own held-out test set):

  | | LSD | MAE |
  |---|---:|---:|
  | S2RNF (simulated HRTF in) | 3.0% | 4.8% |
  | Our geometry (mesh features in, no simulation) | 3.1% | 3.7% |
  | Our simulation → z_meas ridge | 6.4% | 7.8% |

  Not like for like:
  - 16 against 20 kHz. Our gains are smallest at 10–16 kHz (development: 2.5–3.4%), so 16–20 kHz would probably dilute ours.
  - HEALPix (interpolated) against the native 793 directions.
  - Different SOFA versions.
  - Different test subjects.
  - Our AE was pre-trained on further SONICOM subjects (and their simulations) and three other databases; S2RNF used
    160 pairs.

### 2026-10-15 — Why HUTUBS has the largest autoencoder error: the region below −45° that only HUTUBS measured

Prompted by checking each database's measurement grid from the raw files (no model run).

| Measured directions | Count | Elevation | Below −45° |
|---|---:|---|---:|
| SONICOM measured; SONICOM simulated (same grid) | 793 | −45° to 90° | none |
| SYMARE measured; SYMARE BEM (same grid) | 393 | −45° to 90° | none |
| CAR Lab, all 218 subjects (five grids, 392–475 directions; CL126 only to +80°) | 392–475 | −45° to 90° | none |
| **HUTUBS measured** | 440 | **−90° to 90°** | **64** |

On the HEALPix grid, 420 of 3072 pixel centres lie below −45° (13.7%). The figure of 14.6% used before
(§9C 5) is the fraction of the sphere's *area* below −45°, (1 − sin 45°)/2; a ring of 60 pixels sits at
−44.99° and is kept by the elevation ≥ −45° mask. For SONICOM, SYMARE and CAR Lab, the STPS extrapolates the
420 lower pixels from the measured rings above them; for HUTUBS they are interpolated from real measurements.

**Autoencoder error split by region.** LSD is a mean over directions, so the full-sphere value decomposes
exactly: LSD(< −45°) = (3072 · LSD(all) − 2652 · LSD(≥ −45°)) / 420. Values from
`results/ae_lsd_report.json` (`notebooks/AE_LSD_Report.ipynb`, 10-12); nothing re-run.

| FFM26 test, LSD (dB) | AE, all | AE, ≥ −45° | AE, < −45° | Mean, ≥ −45° | Mean, < −45° | AE / mean, ≥ −45° | AE / mean, < −45° |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAR Lab | 1.152 | 1.239 | 0.604 | 4.546 | 4.901 | 0.27 | 0.12 |
| **HUTUBS** | **1.633** | **1.516** | **2.369** | 4.378 | 5.098 | **0.35** | **0.46** |
| SONICOM | 1.256 | 1.367 | 0.556 | 4.602 | 4.796 | 0.30 | 0.12 |
| SYMARE | 1.153 | 1.242 | 0.592 | 4.592 | 5.584 | 0.27 | 0.11 |

FFM26 validation, FFM48k_meas and FFM48k_measim validation give the same picture: the AE error below −45° is
2.33–2.40 dB for HUTUBS against 0.56–0.68 dB for the other three.

- **Below −45° the other three databases are easy and HUTUBS is not.** The extrapolated region still differs
  between people (the mean-HRTF predictor errs 4.8–5.6 dB there), but it is a smooth continuation of the lowest
  measured ring, and the AE reproduces it at about a tenth of the mean's error (0.11–0.13). HUTUBS's lower
  region is measured, with real spectral structure, and the AE reaches only 0.41–0.47 of the mean's error.
  The full-sphere comparison between databases therefore mixes measured and extrapolated regions.
- **This accounts for about half of HUTUBS's excess.**
  - Full sphere, FFM26 test: HUTUBS 1.633 dB LSD against 1.187 for the average of the other three (+0.45 dB).
  - Elevation ≥ −45°: 1.516 against 1.283 (+0.23 dB). Validation shows the same halving (+0.44 → +0.23 dB).
- **The other half remains above −45°.** There HUTUBS still has the largest error, both absolute and relative
  to its own mean (0.32–0.35 of the mean, against 0.27–0.30 for the others), so it is not explained by a larger
  between-person spread alone.
  - Candidate causes, not tested: the fewest training subjects (58); a different measurement setup.
  - Not the 48 kHz upsampling: FFM26 at 44.1 kHz shows the same.
- **Metric caveat.** Shown in LSD, the only per-region metric computed. The training loss and the reported pooled
  RMSE are full-sphere, so HUTUBS's higher RMSE (e.g. FFM26 test 1.79 against 1.25–1.43 dB) includes this
  region effect.

**For the thesis (AE section).** When the per-database errors in `tab:ae` / `tab:ae48k` are discussed, explain
HUTUBS's higher error this way, with the ≥ −45° comparison. This also bears on §9C 5: masking the extrapolated
lower region in evaluation would compare the databases on the region all of them measured.

### 2026-10-16 — Fine-tuning the decoder on the prediction task: feasibility and design *(registered before any code)*

**Question** (the supervisor's suggestion). In mesh → z → decoder → HRTF the decoder is frozen and the regression is
fitted on z. Does adapting the decoder to the prediction task, with the loss on the measured HRTF, change the geometry
result? Put differently: is the frozen decoder, or the latent route, what limits mesh → HRTF?

**Expectation, written before any run: a small gain, mostly not from geometry.**
- The decoder can represent individuals. AE reconstruction is about 1.3 dB, against about 4.25 dB for the geometry
  prediction, and the true z projected on the training PCs reaches R² 0.76 (§2C 10-06 later).
- The route through z loses nothing predictable. Task 1's direct DTF regression, with no autoencoder, gives the same
  R² as the z route on SONICOM (0.069 both).
- Nonlinear models did not beat ridge: MLP, PointNet and the whole-mesh graph network.
- What fine-tuning can recover is limited:
  - the decoder's nonlinearity: decoding the mean z is 0.02 dB worse than the mean HRTF (4.819 against 4.796 dB,
    §2C 10-06 later);
  - the mismatch between a loss on z and an error in dB.
- Expected: within about 1 percentage point of the frozen-decoder result (development: −4.0% LSD), most of it also
  reached with a constant input.

**Design** (`mesh2z/FFM_3_9_DecoderFineTune.ipynb`, to be written; Colab):
- **Subjects and folds:** FFM_3_8's 164 SONICOM development subjects and FFM_3_2_1's folds. The 36 locked subjects
  were spent on 10-12, and the other SONICOM test subjects have no mesh. The result is therefore development-only and
  exploratory: no new held-out number exists for it.
- **Model:** `FFM48k_measim`, as in FFM_3_8.
  - Inputs ẑ: each fold's FFM_3_8 predictions (geometry; the subject's own simulation), refitted inside the fold
    exactly as there.
  - The decoder is fine-tuned on the training subjects of each fold only, so at test time it is fixed.
  - **Trainable: only the decoder FiLM row of the measured SONICOM set (its per-frequency γ and β).** The notebook
    prints the count. The other databases' rows and all other weights stay frozen.
  - Not done: a full-decoder fine-tune (about 260 training ears against ~10⁸ weights), and per-subject (test-time)
    fine-tuning (a new subject has no HRTF to adapt to).
- **Training:**
  - Loss: MSE in dB on the measured HRTF, over bins 1–85 at elevation ≥ −45° (the region of the primary metric).
  - Learning rate and maximum epochs fixed in the notebook header before the run.
  - Early stopping on an inner validation split of each training fold (20% of its subjects), by LSD.
- **Rows** (all out of fold):

  | Row | Input | Decoder |
  |---|---|---|
  | R0 | — (training-mean HRTF) | — |
  | R0z | training-mean z | frozen |
  | R1 | geometry ẑ (= FFM_3_8 geometry row) | frozen |
  | R2 | training-mean z (constant) | fine-tuned |
  | R3 | geometry ẑ | fine-tuned |
  | R4 *(secondary)* | own-simulation ẑ (= FFM_3_8 simulation row) | fine-tuned |

- **Gate:** R0 and R1 reproduce FFM_3_8's development rows (LSD ≥ −45°: 4.558 and 4.378 dB).
- **Metrics:** as FFM_3_8 — LSD on bins 1–85 at elevation ≥ −45° (primary), MAE, full-sphere LSD, pooled RMSE.
- **Primary statistic — the geometry contribution with and without fine-tuning:**
  Δ = [LSD(R3) − LSD(R2)] − [LSD(R1) − LSD(R0)], paired subject bootstrap (2000).
  - R2 is the control that matters. A fine-tuned decoder can lower the error with a constant input alone, by
    producing a better mean. Comparing R3 with R0 would credit geometry with that gain.
- **Reading, fixed now:**
  - |Δ| below 1% of the mean's LSD (≈ 0.046 dB), or its CI including 0: the frozen decoder is not what limits
    mesh → HRTF.
  - Δ ≤ −0.046 dB with its CI below 0: fine-tuning adds geometry information, and the frozen-decoder results
    understate the geometry route.
  - R2 − R0 (what fine-tuning gives a constant input) and R4 (the S2RNF-like route) are reported descriptively.
- **Caveats stated in advance:**
  - The AE was pre-trained on these subjects' measured HRTFs, not their geometry (as in FFM_3_8).
  - The FiLM row is re-estimated in every fold.
  - One run per fold, no seed averaging, unless the spread across folds calls for it.

*Written the same day, before any run* (`mesh2z/FFM_3_9_DecoderFineTune.ipynb`).
- The cells for the model, the regression engine, the data and the development set are copied verbatim from FFM_3_8.
- The gate also checks the refitted ẑ against FFM_3_8's saved `dev_oof.npz` (maximum relative difference < 1e-4).
- The FiLM row is fine-tuned as out' = e^δ·(out − β) + β + δβ on the frozen decoder's output. δ and δβ are per bin and
  start at 0; bin 0 is outside the loss and stays 0.
- Settings fixed in the header: full-batch Adam, learning rate 1e-2, at most 3000 steps, patience 200 steps on the
  inner validation LSD. The same inner split is used for R2, R3 and R4 within a fold.
- A local synthetic check recovered a known per-bin gain and offset (to 0.004 and 0.02) and stopped early as intended.
  No real data were read locally.

*Amendment, 10-16, before any run* (the notebook as committed in e9d11be has not been run):

1. **Leak fixed: cross-fitted inputs for the training rows.**
   - *The leak.* In outer fold f, the training rows' ẑ were their outer out-of-fold predictions. Those come from ridge
     models fitted on the other folds, which include fold f's held-out subjects. Their targets therefore reached the
     fine-tuning inputs indirectly.
   - *The fix.* Within fold f, the training rows get ẑ by cross-fitting inside the training subjects only: 5 groups,
     each predicted by kernel ridge fitted on the other four, with the α already chosen for fold f (by inner CV on its
     training subjects) and fold-internal standardisation on the fitting rows. The constant input of R2 is cross-fitted
     the same way: the mean z of the other four groups.
   - Held-out rows keep the ridge fitted on all of fold f's training subjects (as before).
   - The effect on the 170-parameter FiLM row would be small, but it grows with capacity.
2. **Secondary variants added; the primary (variant A, FiLM row) is unchanged.**
   - **B — all decoder bias vectors** (BitFit, ≈ 9.9 × 10⁴ values), plus A's δ and δβ.
   - **D — the whole decoder** (≈ 5.05 × 10⁷ weights), plus A's δ and δβ. The other FiLM rows and the encoder stay frozen.
   - Each variant has rows R2 (constant) and R3 (geometry), the same Δ = [R3 − R2] − [R1 − R0] and the same reading
     (threshold 1% of the mean's LSD). R4 (simulation) is run for A only.
3. **Training of B and D** (fixed now):
   - Minibatches of 8 ears; Adam with separate learning rates: δ and δβ 1e-2 (as A), bias vectors 1e-3 (B), all decoder
     weights 1e-5 (D).
   - D adds an L2-SP penalty 1e-3 · Σ(θ − θ₀)² towards the pretrained weights.
   - Early stopping on the inner validation LSD, evaluated each epoch: B at most 100 epochs, patience 10; D at most 50,
     patience 5. The best state is restored.
   - Same inner split, loss region (bins 1–85, elevation ≥ −45°) and outer folds as A.
4. **Expectation for B and D** (written now): no geometry gain beyond A. The more capacity, the closer R2 comes to the
   mean HRTF (R0), and the earlier early stopping fires. The earlier nonlinear heads (MLP, PointNet, graph network) all
   overfitted at this N.
5. **Notebook updated to match**, still not run. Local unit tests on synthetic data only (no real data read):
   - When the held-out subjects' targets are changed, the cross-fitted training inputs stay identical. Under the old
     construction they move by up to 6.6.
   - The inner split is by subject, with no overlap.
   - B changes only bias vectors and D all weights. The other FiLM rows and the original decoder are untouched, and the
     inner validation error falls with early stopping and best-state restore working.

### 2026-10-16 (later) — FFM_3_9 result: fine-tuning the decoder does not change the geometry result

`mesh2z/FFM_3_9_DecoderFineTune.ipynb` (commit 64c7de7) run on Colab as registered. Fine-tuning took 3268 s. Outputs:
`mesh2z/output/mesh2z_48k_ft_v1/` (`ft_results.json`, `ft_params.npz`, `ft_results.png`).
- The first run stopped at the save step of Cell 9: Cell 8 had overwritten the output-path variable (fixed in 904d8b4).
- The path was reset and Cell 9 alone re-run. It recomputes from the stored predictions with a fixed seed, so the saved
  numbers equal the printed ones.

**Gate and inputs.**
- The refitted ẑ equals FFM_3_8's `dev_oof.npz` (maximum relative difference 5.8e-8 for geometry, 4.2e-8 for
  simulation).
- R0, R1 and the simulation row reproduce FFM_3_8 to 1e-6 (4.558416, 4.377719, 4.316483 dB).
- Cross-fitted training inputs, R²(z) on the training rows: geometry 0.051–0.061, simulation 0.072–0.080, constant −0.010
  to −0.013. They are slightly below the outer 0.069 and 0.091, as expected from fits on four fifths of the training
  subjects.

**Rows** (development, 164 subjects, out of fold; LSD at elevation ≥ −45°, bins 1–85). Frozen references:
- R0, training-mean HRTF: 4.558
- R0z, decoded training-mean z: 4.574
- R1, geometry: 4.378
- simulation: 4.316

| Fine-tuned | A: SONICOM FiLM row (170) | B: + all decoder biases (99 067) | D: + all decoder weights (50 487 483) |
|---|---:|---:|---:|
| R2, constant z | 4.573 | 4.571 | 4.573 |
| R3, geometry | 4.377 | 4.372 | 4.380 |
| R4, own simulation | 4.316 | — | — |

**Primary statistic** Δ = [LSD(R3) − LSD(R2)] − [LSD(R1) − LSD(R0)], paired subject bootstrap; threshold 0.0456 dB:

| | Geometry contribution, frozen | Geometry contribution, fine-tuned | Δ, 95% CI | Reading fixed beforehand |
|---|---:|---:|---|---|
| **A (primary)** | −0.181 | −0.196 | **−0.016 [−0.020, −0.011]** | the frozen decoder is not the limit |
| B | −0.181 | −0.199 | −0.019 [−0.024, −0.013] | not the limit |
| D | −0.181 | −0.194 | −0.013 [−0.020, −0.006] | not the limit |

**What Δ is made of** (Δ = [R3 − R1] − [R2 − R0]):
- **What fine-tuning does to the geometry prediction itself, R3 − R1:**
  - A −0.001 dB [−0.003, +0.001];
  - B −0.006 [−0.010, −0.001];
  - D +0.002 [−0.003, +0.008].

  At most 0.1% of the mean's LSD.
- **R2 − R0 is +0.015 / +0.013 / +0.015 dB.**
  - The fine-tuned constant input stays where decoding the mean z puts it (R0z − R0 = +0.015 [+0.011, +0.019]); it does
    not reach the mean HRTF.
  - So Δ is mostly that decoded-mean offset. It enters the fine-tuned contribution, measured from R2, but not the frozen
    one, measured from R0.
  - Measured from the decoded mean (R0z), the frozen geometry contribution is −0.196 dB, the same as A's fine-tuned one.
- **Simulation:** fine-tuned minus frozen, −0.001 dB [−0.002, +0.001].

**How far the fine-tunes moved** (inner-validation diagnostics; no fold was still improving at its limit):
- **A** stopped after 0–2 full-batch steps in nine folds and after 11 in one.
  - The learned gain stays within ±2% (max |δ| 0.020) and the offset within 0.11 dB.
  - Inner-validation gain was at most 0.004 dB.
  - With Adam, one step at learning rate 1e-2 already reaches or passes the optimum, so a finer rate could only refine
    within that range.
- **B** moved in 8 (R3) and 7 (R2) of 10 folds (best epoch up to 21 for R3, 42 for R2), with inner-validation gains up to
  0.010 dB.
- **D** moved in only 4 of 10 folds (best epoch 0–6); elsewhere the pretrained decoder was kept.
- The more capacity, the sooner the held-out error rises. These are conservative fine-tunes under the registered
  settings, not every possible schedule, but no variant gained more than 0.01 dB on inner validation.

**Conclusion.**
- By the rule fixed beforehand, the frozen decoder is not what limits mesh → HRTF.
- Fine-tuning the SONICOM FiLM row, every decoder bias, or the whole decoder changes the geometry prediction by at most
  0.006 dB and the simulation prediction by 0.001 dB. This was the expectation registered beforehand: the route through
  z loses nothing that a direct regression keeps, and nonlinear heads did not help earlier.
- Development set only and exploratory; no held-out SONICOM subjects with meshes remain.
- **For the thesis**, one paragraph in the mesh section answers "is the latent or the decoder the bottleneck?" with A's Δ,
  the R3 − R1 values and the decoded-mean decomposition.

**Thesis note: state the bottleneck clearly** (decided 10-16). The geometry chapter must say plainly what limits
geometry → measured HRTF, what has been ruled out, and what remains open. All numbers are already in this log.

*Ruled out — each tested and not the limit:*

| Candidate | Test | Evidence |
|---|---|---|
| The latent route / the frozen decoder | FFM_3_9; Task 1 | Fine-tuning the FiLM row, the biases or the whole decoder changes the geometry prediction by ≤ 0.006 dB; direct DTF regression without the AE gives the same R² as the z route (0.069) |
| Stronger regressors (among those tested) | FFM_3_3, 3_5, 3_6, 3_7 | None beats ridge: MLP 0.156 dB worse with the same output space and weighting; PointNet worse; whole-mesh graph network 3.7% of the gap, then overfits |
| Point sampling, descriptor size, orientation | FFM_3_6 | Exact per-triangle features = 20 000-point features within 0.003 dB; M = 1024 and surface normals add nothing |
| Pinna-only geometry, pinna alignment | FFM_3_4 | Both worse than the 45 mm crop with head measures |
| The torso missing from the simulations | Task 1b | Adding the torso (SYMARE HTE) makes the simulation predict the measurement slightly worse (−0.017 [−0.036, −0.002]) |
| Sampling rate and data faults | FFM_3_8 at 48 kHz | Geometry result unchanged: 3.6% below the 164-subject mean at 48 kHz, 3.5% at 44.1 kHz (pooled RMSE) |
| The head beyond the ear crops *(added 10-23)* | FFM_3_10 (development) | Adding it changes the LSD by +0.003 dB [−0.004, +0.009]; it raises the low-band R² by 0.089 for the simulated latent but by 0.008 for the measured one |

*The bottleneck, by band:*
- **Below 5 kHz — the measured HRTF does not follow the scanned shape (Task 1 cause B).**
  - The features explain half of the simulated HRTF but a tenth of the measured one (R² 0.50 vs 0.12, SONICOM).
  - The shape-driven change is present in the measurement at nearly full size (Task 2: 0.79–1.11 × the simulated
    response, same pattern).
  - But the measured low band carries about three times the simulated between-person variance (3.1 × [2.9, 3.4]
    SONICOM, 3.5 × [3.0, 4.1] SYMARE), which shape does not explain.
- **Above 6 kHz — the shape → HRTF relation itself is not learned from these features at this N (cause A).**
  - Even the noise-free simulation is poorly predicted (R² ≤ 0.23 SONICOM, ≤ 0.08 SYMARE).
  - The measured values are lower again.
- **Whole band — simulation and measurement disagree on individuals.**
  - A person's own BEM predicts their measured deviation worse than the mean (DTF R² −0.50); per-side correlation is
    0.16–0.23 on MRI and optical meshes.
  - Used as an input, the own simulation explains R² 0.09 (SONICOM, z and DTF) and 0.08 / 0.14 (SYMARE, z / DTF).

*Net result to report:*
- Detectable, not usable. On the held-out test, geometry is 3.1% below the mean in LSD (−0.134 dB [−0.204, −0.068]) and
  the own simulation 6.4% (FFM_3_8).
- Independent check: S2RNF, a different method, gives 3.0% on its own test.

*Open, untested candidates for the extra low-band variance — name them as open, not as findings:*
- scan versus measurement posture (head–torso angle, shoulders);
- the low-frequency measurement side (equalisation, compensation, positioning);
- per-ear measurement repeatability (no repeated measurements of the same ears are held; §9C 4c).

*Wording limits* (§2C 10-07):
- The result concerns these representations, SONICOM's preprocessed scans, simulation-defined canal positions, magnitude
  only and N = 164 / 46. Say "within these conditions", not "geometry cannot".
- "The regressor is not the limit" means: stronger regressors among those tested did not help.

### 2026-10-17 — Thesis: comparison with published work, Chapter 2 restructured, outstanding revisions

**Done today (Overleaf and local copy in sync).**
- **`tab:sota` (Chapter 4, `sec:res-sota`).** Measurement-free HRTF prediction in published work against this thesis.
  - Every number was checked in the paper itself; sources are in
    `Thesis/notes/SOTA_comparison_overleaf_guide_2026-10-17.md` §7.
  - Only S2RNF and this work report a population-mean baseline on measured targets.
  - GISNO (Interspeech 2026, 2.92 dB LSD on HUTUBS) does not state whether its targets are measured or simulated, and
    names measured HRTFs as future work. An e-mail asking the corresponding author was sent on 10-17.
  - BEM used directly is worse than its reference in all three studies that report one: this work, S2RNF, and Pirard
    et al. 2026 (9.0 dB, against 6.6 dB for a random other subject's measured HRTF).
  - Zhang et al. 2021's 4.80 dB is a mean absolute error, which Zhao et al. 2024 tabulate as LSD.
- **Chapter 2, "why the numbers cannot be compared directly".** The paragraph in `sec:bg-metrics` notes that there is
  no common benchmark. Its examples are the survey count (3 of 76 studies report an average-HRTF baseline), the
  inconsistent metric names and the 5.05 → 4.73 dB protocol shift.
- **Chapter 2 split into two parts** (Craig: background and literature review are both needed; the template has a
  single "Background" chapter).
  - A, background: §2.1–2.5.
  - B, literature review in RQ order: §2.6 pooling (RQ1), §2.7 measurement-free prediction (RQ2), §2.8 simulated vs
    measured (RQ3), §2.9 morphology (RQ4), §2.10 neural BEM surrogates, §2.11 evaluation practice, §2.12 Gaps 1–5.
  - The database table now states that only HUTUBS is measured below −45° (−90° to 90°), and the 48 kHz convention.
  - Local file written; to be pasted into Overleaf.

**Outstanding revisions** (by priority; tick off as done):

*A. Content and facts*
- [ ] 1. **FFM_3_9 (decoder fine-tuning, Craig's suggestion) is not in the thesis at all.**
  - Chapter 4, RQ3: add it to the "ruled-out" table (from the 10-16 thesis note), with a table generated from
    `ft_results.json` and one paragraph.
  - Chapter 3: a short design paragraph (cross-fitting, the three scopes, the constant-input control).
  - Appendix A: a notebook row.
- [ ] 2. Statement item 5: "not learnable from 164 subjects" → "not learned from these features and 164 subjects"
  (wording limit, 10-16).
- [ ] 3. Chapter 4 opening paragraph still reads "every geometry-to-latent experiment in the order they were carried
  out".
- [ ] 4. `tab:geometry-summary` ("Every geometry-to-latent attempt") lacks the 48 kHz test and FFM_3_9. Add the rows,
  or restrict the caption to the 44.1 kHz development experiments.
- [ ] 5. `tab:mesh-ridge`'s 6.5% is measured against the 339-subject pool mean, `tab:mesh48k`'s against the
  164-subject mean. State which mean in both captions.
- [ ] 6. Why HUTUBS reconstructs worst (10-15): not yet in the AE section.
- [ ] 7. The 44.1 kHz (FFM26) tables do not state the sample rate in their captions: landmarks, mesh-ridge, PointNet,
  physics, within-person, pinna. Either add "(FFM26, 44.1 kHz)" or put one sentence at the start of the mesh section.
- [ ] 8. Glossary:
  - `z` says "93 frequency bins"; add 86 for the 48 kHz models;
  - add ITD / ILD, MAE, SD / SDE (used in the `tab:sota` notes) and KEMAR.
- [ ] 9. Chapter 5 `%` notes still say three RQs, 6.5% and "four databases". Update to RQ1–RQ4 and the held-out
  numbers (3.1% / 6.4%).
- [ ] 9b. Chapter 1 Thesis Organisation, Chapter 2 bullet: rewrite for the two-part chapter (text given in chat,
  10-17).
- [x] 9c. *(10-17: seven entries cleaned in `Thesis/notes/Citation_cleaned_2026-10-17.bib`, added to the local bib, keys
  written into the Chapter 2 notes; `brinkmann2019cross` and `perez2018film` repaired. Paste into Overleaf.)*
  Chapter 2 bibliography still missing; export from Zotero, do not type by hand:
  - Andreopoulou, Begault & Katz 2015;
  - Ziegelwanger, Majdak & Kreuzer 2015 (JASA);
  - Poole & Picinali 2026 (arXiv:2608.16722);
  - Dinakaran et al. 2018;
  - Zolfaghari et al. 2016 and 2017;
  - Tew et al. 2012.

*B. Structure*
- [ ] 10. Chapter 4 reordered by RQ (agreed 10-16):
  - 4.1 RQ1: AE results, HUTUBS region, disentanglement as a pilot;
  - 4.2 RQ2: landmark pilot, mesh development, held-out test, `tab:sota`;
  - 4.3 RQ3: ruled-out candidates including FFM_3_9, physics ceiling, Task 1, torso, within-person;
  - 4.4 RQ4: Task 2;
  - 4.5 Discussion answering RQ1–RQ4.
- [ ] 11. Chapter 3 mirrored to the same RQ order (10 `\todo` left).

*C. Length and format*
- [ ] 12. **Body (Chapters 1–5) is 33 pages; the requirement is ≥ 40 excluding appendices.** Chapter 2 (planned 9–11
  pages, about 2 now) and Chapter 3 will supply the difference, so write them first.
- [ ] 13. Abstract is about 359 words; the template asks for 200–300.
- [ ] 14. Appendix A: 3 `\todo` left.

*D. Waiting*
- [x] 15. Reply from the GISNO corresponding author. When it arrives, update `tab:sota`'s GISNO row, note a, and the
  GISNO sentence in `sec:res-sota`. *(Done 10-27: §2C 10-27.)*

### 2026-10-23 — FFM_3_10: the head mesh outside the ear crops added to the geometry: design *(registered before any code)*

**Question** (Harry). The geometry inputs come mostly from around the ears, while the low frequencies are governed by
the head. On the 164 SONICOM development subjects, does adding the head mesh outside both ear crops (REST) to the
current geometry (nine measures + ear-crop embedding, GEO) improve the prediction of the measured HRTF? mesh → latent
was closed on 10-05 and reopened at 48 kHz on 10-12 (FFM_3_8); this is one further, specific question.

**Not a repeat of FFM_3_2_1's "ear + head"** (CV-R² 0.0669 against 0.0671 for the ear alone): there the head sample
contained the ear crops, its five blocks weighed as much as the five ear blocks, and it was not combined with the
measures. Here the sources do not overlap, every source has the same weight (FFM_3_8 `source_kernel`), and REST's
weight is chosen inside the training folds (row R5).

**What is known, and the expectation, written before any run:**
- Task 1: at 0.2–5 kHz the same geometry explains 0.50 of the simulated latent but 0.12 of the measured one. Task 2:
  the measured low band has 3.1 × [2.9, 3.4] the simulated between-person variance, while the shape-driven responses
  carry over at 0.79–1.11 ×. If the measured HRTF = the simulated shape part + an independent rest, only about 1/3.1
  of a gain on the simulated low band can reach the measured one, and complete shape information would bound the
  measured low-band R² at about 0.32.
- The low band is 25 of the 85 bins and has the smallest errors. From FFM_3_8's error curve, a low-band R² gain of
  +0.04 moves the full-band LSD by about −0.008 dB and +0.20 by about −0.040 dB, both below the 0.046 dB threshold.
  The primary test therefore detects only a gain of at least 1% over the whole band; a low-band gain is read from the
  secondary statistic.
- Expected: full-band ΔLSD within ±0.01 dB; measured low-band ΔR² below +0.02; the simulated target may gain more.

**Design** (`mesh2z/FFM_3_10_Mesh2Z_HeadRest.ipynb`; Cells 2–6 copied verbatim from FFM_3_8; Colab):
- **Subjects, folds, encoder, engine:** as FFM_3_8 — 164 development subjects, FFM_3_2_1's folds, `FFM48k_measim`
  frozen, kernel ridge with fold-internal standardisation, α by inner 5-fold CV. Development only: the 36 locked
  subjects were spent on 10-12 and are not read. The result is exploratory; there is no held-out number for it.
- **Sources:**
  - GEO: FFM_3_8 `geometry_blocks()`, unchanged.
  - REST: all 50 000 `head_pts`, kept if z > −75 mm and farther than 45 mm from both ear canals, first 20 000 of
    those; head frame; for right-ear rows the whole head mirrored (y → −y); RFF mean embedding at GEO's five scales
    (σ 0.005–0.1 mm⁻¹, the same random directions).
  - HEADM: five head measures from the head points with z > −75 mm — width (y range at |z| < 20 mm, 0.5–99.5th
    percentile), back (−x, 0.5th percentile), front (x, 99.5th percentile), convex-hull volume, and the canal's x
    minus the median head x (each row its own canal).
- **Rows** (each source unit-scaled, sources added):

  | Row | Input | Role |
  |---|---|---|
  | R0 | training mean | baseline |
  | R1 | GEO | gate: reproduces FFM_3_8's development row (LSD 4.377719 dB, CV-R² 0.069439) |
  | **R5** | GEO + w·REST; w ∈ {0, 0.25, 0.5, 1, 2} and α chosen jointly by the inner CV of each outer training fold | **primary** |
  | R2 | GEO + REST (w = 1) | secondary |
  | R3 | GEO + HEADM | secondary |
  | R4 | GEO + REST + HEADM | secondary |
  | R6 / R7 | REST only / HEADM only | descriptive |
  | C1 | R5 with REST permuted across subjects (both ears together), 100 permutations (seeds 0–99), w and α re-chosen each time | null for the increment |

- **Targets:** the measured latent (primary); the simulated latent (SONICOM_SIM through `FFM48k_measim`, as Task 1)
  as a diagnostic, with the same rows and rules. The DTF targets of the plan are dropped: in Task 1 the DTF and the
  latent routes agree.
- **Bands:** Task 1's, limits included — 0.2–5 kHz (bins 2–26), 6–12 kHz (bins 32–64), 12–16 kHz (bins 64–85);
  5–6 kHz (bins 27–31, in none of Task 1's bands) reported descriptively. Full-band statistics use all bins.

**Primary statistic:** ΔLSD = LSD(R5) − LSD(R1), LSD on bins 1–85 at elevation ≥ −45° (per subject, the mean of both
ears); paired subject bootstrap (2000) and one-sided paired t, and C1's null of the same difference,
p_C1 = (1 + #{C1 ≤ observed}) / 101.

**Reading, fixed now:**
- ΔLSD ≤ −1% of R0's LSD (≈ −0.046 dB), its CI below 0, and p_C1 ≤ 0.05: the head beyond the ear crops improves the
  prediction of the measured HRTF.
- Otherwise: adding REST does not improve the full-band prediction by 1% or more. This alone does not say that REST
  carries no low-frequency information; that is read from the secondary statistic.

**Secondary, registered:**
- ΔR²_low = CV-R²(R5) − CV-R²(R1) on the measured latent, 0.2–5 kHz columns (SSE and SST summed over subjects); subject
  bootstrap CI and C1's null. Read as "REST adds low-band information" only if ≥ +0.02, its CI above 0, and above
  C1's 95th percentile.
- Ratio ΔR²_low,meas / ΔR²_low,sim (R5 on each target, the same bootstrap draws). Predicted ≈ 1/3.1 ≈ 0.32;
  interpreted only if the CI of ΔR²_low,sim is above 0 — a CI containing 0.32 supports the decomposition above, one
  clearly above 0.32 contradicts it.
- Descriptive: per-band R² and LSD of every row on both targets, the w chosen in each fold, REST point counts and the
  HEADM distributions.

**Caveats stated in advance:** SONICOM's processed scans may have filled surfaces on top of the head (hair, holes); the
height distribution of the REST points is recorded and outliers are listed. The canal positions are those of the
simulation receivers, as in every mesh experiment. Run for the final thesis (11-06), not the draft.

### 2026-10-23 (later) — FFM_3_10 result: the head beyond the ear crops does not improve the prediction

`mesh2z/FFM_3_10_Mesh2Z_HeadRest.ipynb` run on Colab as registered, the same day. C1 (100 permutations) took 6.4 min.
Outputs: `mesh2z/output/mesh2z_48k_rest_v1/results.json`, `per_band.png` (`per_subject_lsd.npz` kept locally).

**Gates passed.** R0 and R1 reproduce FFM_3_8's development rows (LSD 4.558416 and 4.377719 dB, CV-R² 0.069439);
with w fixed at 0 the R5 code reproduces R1; R1's 0.2–5 kHz CV-R² reproduces Task 1 (0.1234 measured, 0.4955
simulated).

**Features.** REST kept 33 961–41 606 points per subject (median 37 785) and used 20 000 for every subject. One outlier in
head-top height, SON_P0181 (99.9th percentile 127.6 mm against a median of 145.1 mm); kept, as registered.

**Primary (registered):** ΔLSD = LSD(R5) − LSD(R1) = **+0.0027 dB [−0.0038, +0.0088]**, one-sided t p = 0.79,
p_C1 = 0.95; threshold −0.0456 dB. **Reading: adding the head beyond the ear crops does not improve the full-band
prediction by 1% or more** — the point estimate is slightly worse.
- R5 chose w = 0.25 in 7 of the 10 outer folds and w = 0 in 3 (simulated target: 0.25 in 9 of 10). With REST
  permuted (C1), w > 0 was chosen in 11 of 1000 folds, so C1's ΔLSD is exactly 0 in most permutations.

**Secondary (registered):**
- ΔR²_low (measured latent, 0.2–5 kHz) = **+0.0075 [+0.0019, +0.0133]**, above all 100 C1 permutations (p_C1 = 0.010;
  C1 95th percentile 0.0000), but below the registered +0.02. By the rule: no registered evidence that REST adds
  low-band information. It is a detectable signal, too small to count.
- ΔR²_low (simulated latent) = **+0.089 [+0.075, +0.103]**: 0.496 → 0.584.
- Ratio measured / simulated = **0.084 [0.022, 0.152]**, below the predicted 0.32. The registered reading covered only
  a CI containing 0.32 (supports the decomposition) or clearly above it (contradicts it); a CI below 0.32 is
  descriptive.

**Descriptive** (CV-R²(z), all / 0.2–5 / 5–6 / 6–12 / 12–16 kHz; LSD ≥ −45° in dB, all and 0.2–5 kHz):

| Row | Measured R² | Measured LSD | Simulated R² | Simulated LSD |
|---|---|---|---|---|
| R0 training mean | — | 4.558 / 2.363 | — | 4.078 / 1.114 |
| R1 GEO | 0.069 / 0.123 / 0.169 / 0.071 / 0.047 | 4.378 / 2.170 | 0.196 / 0.496 / 0.456 / 0.226 / 0.127 | 3.596 / 0.733 |
| R5 GEO + w·REST | 0.069 / 0.131 / 0.170 / 0.069 / 0.047 | 4.380 / 2.162 | 0.198 / 0.584 / 0.478 / 0.224 / 0.126 | 3.611 / 0.706 |
| R2 GEO + REST | 0.066 / 0.131 / 0.164 / 0.062 / 0.047 | 4.393 / 2.169 | 0.175 / 0.563 / 0.443 / 0.195 / 0.109 | 3.687 / 0.744 |
| R3 GEO + HEADM | 0.061 / 0.124 / 0.159 / 0.059 / 0.039 | 4.402 / 2.174 | 0.194 / 0.556 / 0.470 / 0.222 / 0.122 | 3.605 / 0.714 |
| R4 GEO + REST + HEADM | 0.060 / 0.129 / 0.157 / 0.055 / 0.042 | 4.407 / 2.169 | 0.171 / 0.579 / 0.448 / 0.191 / 0.103 | 3.696 / 0.734 |
| R6 REST only | 0.017 / 0.048 / 0.044 / 0.012 / 0.014 | 4.534 / 2.300 | 0.036 / 0.204 / 0.109 / 0.039 / 0.015 | 4.036 / 1.031 |
| R7 HEADM only | 0.022 / 0.054 / 0.060 / 0.020 / 0.015 | 4.512 / 2.278 | 0.043 / 0.210 / 0.139 / 0.048 / 0.019 | 4.012 / 0.991 |

- R5 − R1 in LSD by band, measured: 0.2–5 kHz −0.008, 5–6 kHz +0.006, 6–12 kHz +0.009, 12–16 kHz −0.001 dB; the
  low-band gain is cancelled by the bands above. Simulated: −0.027, +0.002, +0.026, +0.018 (full band +0.015 dB),
  although the simulated R² over all bins rises (0.196 → 0.198): w is chosen on the latent SSE, which weighs the bands
  differently from an LSD in dB.
- Equal source weights hurt, as on SYMARE in Task 1: against R1, R2 is +0.015 dB [+0.0005, +0.029], R3 +0.024
  [+0.010, +0.038] and R4 +0.030 [+0.011, +0.047] worse.

**What it means.**
1. The head beyond the ear crops does not improve the prediction of the measured HRTF (registered reading). Add it to
   the ruled-out list (thesis note of §2C 10-16 later and `tab:ruled-out`).
2. REST carries head shape that matters for the simulated low band (+0.089 R²), but only about a twelfth of that
   gain appears in the measured latent (+0.0075). The rough expectation registered beforehand, about a third, assumed
   that the measured low band is the simulated shape part plus independent variation. The measured low band follows
   this extra head shape even less than its threefold variance implies, which strengthens cause B below 5 kHz. Why was
   not tested; the open candidates are those of §2C 10-16 later (posture between scan and measurement, the
   low-frequency side of the measurement, repeatability) and the missing torso altering the head's low-frequency
   effect in the measurement.
3. Development only, exploratory; the held-out 3.1% is unchanged.

**Thesis note** (for the final, or the draft if time allows): a row in `tab:ruled-out` — "The head beyond the ear
crops | FFM_3_10 (development) | adding it changes the LSD by +0.003 dB [−0.004, +0.009]; it raises the low-band R² by
0.089 for the simulated latent but by 0.008 for the measured one". Text in `Thesis/notes/Ch4_reference_text_2026-10-22.md` §7.3b.

### 2026-10-25 — Where the SONICOM scans end at the neck (thesis figure and Ch3 text)

For a thesis figure of the input scan (`fig:sonicom-mesh`, 4.2.2, rendered by `Thesis/UnitB/make_mesh_side_view.py`)
and the Ch3 description of the head-sample cut. Read-only, local: `analysis/neck_end_stats.py`
on the 200 SONICOM subjects with meshes — the `PXXXX_preprocessed.stl` scans and `PXXXX_graded_left/right.stl` meshes
in `synthetic/`, and the 50 000 head points per subject in `mesh2z_samples`. Depth below the interaural axis (z = 0), mm:

| | min | median | max |
|---|---:|---:|---:|
| Lowest point, preprocessed scan (the input to the mesh features) | 81.2 | 111.2 | 147.6 |
| Lowest point, graded meshes of the Mesh2HRTF simulations (lower of left/right) | 81.2 | 110.6 | 147.4 |
| Lowest point 30 mm or more behind the axis (x < −30 mm) | 50.8 | 77.5 | 130.0 |

- The first row reproduces §2C 10-11 (81–148 mm, median 111). The graded meshes also end at the neck; 6 of 200 end
  more than 2 mm higher than their scan (at most 21.8 mm).
- **The head-sample cut z > −75 mm (FFM_3_2_1; also REST and HEADM in FFM_3_10) does not give every head the same
  lower boundary.** The curators' neck cut is inclined, higher at the back: 30 mm or more behind the axis, 71 of 200
  scans end above −75 mm, so there the head sample ends where the scan does. FFM_3_2_1's assertion checks only the
  lowest point. The cut removes a median of 11.1% of the head points (1.9–19.7%), from the throat. No result changes;
  the Ch3 reference sentence ("cut … above the variable height at which the scans end") is corrected in
  `Thesis/notes/Ch3_reference_text_2026-10-23.md` §3.2.
- P0010 (the subject of the figure and of `fig:ear-crop`): lowest point 115.5 mm, 73.1 mm behind the axis, 11.3% of its
  head points below the cut.

### 2026-10-27 — GISNO: the population-mean baseline

At our request the GISNO authors scored the mean of their training subjects on their 15 test subjects (personal
communication; thesis Section 4.2, `huang2026personal`). Reproducing that value under their stated protocol:
`analysis/gisno_baseline_check.py`; the effect of the frequency range alone: `analysis/hutubs_band_baseline.py`.
The correspondence itself is not reproduced here.

### 2026-10-27 — Ch3 pipeline figure (`fig:pipeline`): assets for SONICOM P0010

Evaluation only, nothing trained. `notebooks/FFM48k_PipelineFigure_Arrays.ipynb` (Colab, run by Harry) read FFM48k_measim,
the 48 kHz data and FFM_3_8's `dev_oof.npz`, and saved `results/pipeline_fig/pipeline_P0010.npz`: left ear of P0010
(development subject, fold 9) in its own frame — measured, AE reconstruction, z, out-of-fold ẑ (geometry row), decoded
prediction, training mean of the fold. Gate (both ears, bins 1–85, el ≥ −45°): geometry 3.8011 dB (recorded 3.8011),
fold mean 3.8060 (3.8060); AE reconstruction 1.3816 (not gated). Checked locally as well: the measured and mean arrays are
identical to the local data, and ẑ equals `dev_oof` row 14.
- Left ear alone: prediction 3.762, mean 3.692, reconstruction 1.352 dB. P0010's geometry gain is 0.13% over both ears,
  against a development median of 3.9% (P0071), so its predicted map looks like the mean. P0010 was kept for consistency
  with `fig:sonicom-mesh` and `fig:ear-crop`.
- Panels (`Thesis/UnitB/make_pipeline_map_assets.py`, `make_pipeline_mesh_assets.py`, outside the repo; output
  `Thesis/UnitB/pipeline_assets/`): maps at 3.94 and 8.06 kHz without the −45° line, title or colour bar, one shared
  scale per frequency (−39.7 to 5.2 and −56.6 to −1.2 dB); latents z and ẑ (86 × 64, shared ±89.3); head (three-quarter
  and side views) and left-ear crop. Prompt, upload list, label checklist and draft caption:
  `Thesis/notes/fig_pipeline_prompt_2026-10-27.md`.

### 2026-10-28 — SONICOM scan processing and hair: what is known (literature check, no run)

Prompted by the question whether SONICOM's "hair removed" step could bias the mesh inputs of mesh → latent.
- **Acquisition** (Engel et al. 2023, JAES 71(5), open-access PDF): EinScan Pro 2X 2020 (structured light, hand-held),
  EXScan Pro v3.7.0.3. **Subjects wore a wig cap during the scan and, for consistency, during the HRTF measurement.**
  Unscanned patches remain, typically behind the ear and in other narrow concave structures; facial or long hair
  protruding from the cap is not captured (low reflectivity). Point cloud 0.5 mm; minimal filtering and smoothing;
  holes < 10 cm perimeter filled, then a watertight mesh.
- **Processing for the synthetic HRTFs** (Poole et al. 2025, arXiv 2507.05053): Frankfurt-plane alignment, "all head
  and facial hair were removed", truncation below the neck, minimal smoothing, interpolation of missing regions; scans
  with excessive distortion around the pinnae excluded (number not given). **No method (manual or automatic) and no
  accuracy is reported.** Pirard et al. 2026 only restate this.
- **Assessment.** Head hair is matched between scan and measurement (cap in both). Residual risks: interpolated
  surface behind the ear (inside the 45 mm ear crop), facial hair removed from the mesh but present in the measurement,
  and hair/cap acoustics, which no rigid mesh models. Any mismatch between processed mesh and head appears as cause B
  (the simulations use the same processed meshes), not as cause A. SYMARE (MRI meshes, different processing) shows a
  similar simulation–measurement disagreement (r 0.23 vs 0.16; low-band variance ratio 3.5 vs 3.1), so the
  disagreement is not specific to SONICOM's processing (suggestive only). Raw scans (`3DSCAN/`) are not stored
  locally, so the raw-vs-processed difference was not measured.
- **Thesis** (record `Thesis/notes/SONICOM_hair_edits_2026-10-28.md`): Ch2 2.1.2 SONICOM paragraph — wig cap, interpolated
  gaps, hair-removal method and accuracy not reported; Ch4 4.3.5 — hair added to the untested cause-B candidates
  (Treeby, Pan & Paurobally 2007, JASA 122(6):3586–3597: on a spherical head with measured hair impedance, hair
  perturbs the HRTF, ITD and ILD asymmetrically, at the frequencies below 5 kHz examined). New bib `treeby2007effect`.
- For comparison, our own HUTUBS mean over the full band is 4.48 dB LSD (FFM26 test, `results/ae_lsd_report.json`,
  different preprocessing): the band alone moves the mean by more than 1 dB — an example for Ch2 `sec:bg-metrics`.

**Thesis changes** (record: `Thesis/notes/GISNO_reply_edits_2026-10-27.md`; Ch2 2.2.2 file synced): GISNO's target
"measured", range 0.2–8 kHz, mean 6.25 dB shown with a note, **no reduction**; Ch4 `sec:res-sota` explains the check;
Ch2 2.2.2 closing paragraph, Gap 2 and Gap 3, and the Ch4 limits paragraph adjusted; new bib entry `huang2026personal`.
Possible follow-up e-mail: how the mean was formed, the LSD formula, and whether 2.92 dB used the same code.

---

## 4C. Experiment index *(additions to §4B)*

| Script | Question | Headline result |
|---|---|---|
| `notebooks/FFM_2_6_2QuadDB.ipynb` | Can SYMARE join without degrading the other databases? | 774 subjects, val 1.4855 dB; existing partitions move ≤ 0.024 dB |
| `landmark2z/FFM_2_8_Landmark2Z.ipynb` | Do correctly matched landmarks depend on the latent? (N = 61) | dCor²ᵤ 0.0710, p = 0.0020; controls clean |
| `landmark2z/FFM_2_9_RFF_MLP.ipynb` | How much is usable? | Best CV-R² 0.0845; RFF gain within noise; MLPs overfit; stopping rule triggered |
| `landmark2z/FFM_3_0_CapacityStress.ipynb` | Does training longer or adding features rescue it? | No: 5000-epoch held-out R² +0.008; M = 8…512 without trend |
| ad hoc, 09-25 | Does reusing subject 1's grid corrupt HUTUBS/SONICOM? | No: 96/96 and 399/399 identical grids |
| `mesh2z/sample_meshes.py` | Can 200 SONICOM meshes be brought into one checked frame? | 200/200 pass ten checks; six corruptions caught |
| `mesh2z/FFM_3_1_Mesh2Z_Latents.ipynb` | What are the dB bounds for any mesh → latent result? | 1.396 / 2.601 / 4.844 / 4.842 dB; ear averaging costs 1.2 dB |
| `mesh2z/FFM_3_2_1_Mesh2Z_RFF.ipynb` | How much of an individual's latent does the mesh predict? | 6.5% of the gap, 0.23 dB; LSD 4.40 vs 4.65 dB; p = 0.010; signal at 0–6 kHz; stopping rule triggered |
| `mesh2z/physics_ceiling.py` | Does an exact BEM solve from the same scan predict the measured HRTF? | No: R² −0.50 against the mean (DTF); +0.03 optimally shrunk, below ridge's +0.06; identifies the person 11.9% vs 0.6% chance |
| `mesh2z/sim_vs_measured.py` | Does BEM reproduce measured individual deviations, MRI vs optical meshes? | Per side: SYMARE r 0.225, SONICOM 0.164; BEM as predictor R² −0.52 / −0.53; the 08-08 0.756 mixed left and right ears |
| `mesh2z/FFM_3_4_Mesh2Z_PinnaOnly.ipynb` | Pinna only, pinna alignment, and are shape neighbours z neighbours? | Pinna-only best +0.058 < 0.076; paired +0.054 dB worse [+0.029, +0.079]; distance Spearman 0.14–0.18; kNN overlap ~2× chance. **Arm closed** |
| `mesh2z/within_person.py` | Within one person, does the left–right pinna difference predict the left–right latent difference? | Yes but negligibly: CV-R² +0.0046, p = 0.005; pinna and latent differences are 58–67% of between-person |
| `mesh2z/FFM_3_3_Mesh2Z_PointNet.ipynb` *(exploratory)* | Do trained point features beat random features? | No: PointNet xyz +0.14 dB and RFF +0.51 dB worse than ridge, both CIs exclude 0; RFF memorises shuffled targets |
| `mesh2z/FFM_3_5_Geometry2Z_TrainValTest.ipynb` | Does a fixed split with epoch-trained MLPs change either result? | No: MLP below the mean (landmarks −1.4%, meshes −1.1% of the gap) and near the shuffled control; ridge +0.7% / +2.4%; validation 9 / 33 subjects |
| `mesh2z/pack_full_meshes.py`, `mesh2z/FFM_3_6_Mesh2Z_FullMesh.ipynb` | Does the result hold for the complete meshes, without sampling, with matched output space and weighting? | Yes. Exact vs sampled features: Δ ≤ 0.003 dB; ridge 4.649 dB (4.3%); block weighting +0.065 dB over column; M = 1024 and normals add nothing; MLP 0.156 dB worse than ridge. Stopping rule triggered |
| `mesh2z/FFM_3_7_Mesh2Z_DirectFullMesh.ipynb`, `mesh2z/fullmesh_graph.py` | Does a network trained on every face of the whole head do better? | No: best epoch 24, 4.671 dB (3.7%), CI vs mean includes 0; overfits afterwards (probe R² 0.36, val −0.035 at epoch 100); below ridge |
| `preprocess_48k/` (`preprocess_48k.py`, `verify_48k.py`, `pack_48k.py`) | Can every database be brought to 48 kHz with every subject checked individually? | Yes: 1034 subjects; the 44.1 kHz arrays are reproduced exactly by the same code; four data faults found and handled (§5C #23–24); `data_48k_2026-10-10.zip` |
| `notebooks/FFM48k_Train.ipynb` | Does the 48 kHz autoencoder train as FFM26 did, and are the simulations in its distribution? | Val 1.4793 dB, 64.8–73.1% below the baselines; simulations at 0.93 (SONICOM) and 1.40 (SYMARE BEM) × the measured error → `FFM48k_measim` for Task 1 |
| `notebooks/FFM48k_Train.ipynb`, `INCLUDE_SIM = True` (`FFM48k_measim`) | Can the simulations join without costing the measured databases? | Yes: measured pooled 1.4640 vs 1.4793 dB; simulations at 0.71 / 0.95 × their partners' error |
| `task1/Task1_SimVsMeas.ipynb` | Task 1: is geometry → HRTF limited by learnability (A) or by the measured HRTF not following the scan (B)? | Both, by band. Geometry → simulated vs measured CV-R² 0.197 vs 0.069 (SONICOM, CI of the difference excludes 0), 0.074 vs 0.025 (SYMARE); 0.2–5 kHz 0.50 vs 0.12 (B); 6–16 kHz simulated ≤ 0.23 (A); own simulation → measured 0.09 |
| `preprocess_48k/` (`SYMARE_HTE`, `pack_hte.py`), `task1/Task1b_TorsoCheck.ipynb` | Does the missing torso explain Task 1's low-band gap? (SYMARE HTE05kHz vs HE16kHz, 0.2–5 kHz) | No: own simulation → measured R² 0.087 (HTE) vs 0.104 (HE), difference −0.017 [−0.036, −0.002]; ρ 0.21 vs 0.28 |
| `task2/Task2_ShapeModes.ipynb` | Task 2, first version: which shape–HRTF modes (named measures, landmarks) appear in both simulated and measured HRTFs? | All six leading modes survive (cosine 0.27–0.73, p ≤ 0.012) at 0.79–1.11 × the simulated amplitude; SONICOM protrusion mode matches in every band; the measured low band has ~3 × the simulated between-person variance |
| `task2/redraw_task2_figures.py`, `task2/between_person_variance.py` | Thesis-size Task 2 figures; a reproducible source for the between-person variance table | Both reproduce the saved statistics exactly; variance ratio 0.2–5 kHz 3.1 × [2.9, 3.4] / 3.5 × [3.0, 4.1] |
| `notebooks/AE_LSD_Report.ipynb` | Autoencoder reconstruction in LSD, with the baseline, for FFM26 (val, test) and both 48 kHz models (val) | Recorded RMSEs reproduced; FFM26 test LSD 1.299 vs mean 4.604 dB; FFM48k_meas val 1.310 vs 4.770; 48 kHz test not read |
| `mesh2z/FFM_3_8_Mesh2Z_48k.ipynb` | mesh → latent at 48 kHz, scored as S2RNF: how far below the mean, in LSD and MAE (bins 1–85, el ≥ −45°)? | Development: geometry −4.0%, simulation −5.3%, both −5.8% LSD; fold-internal scaling changes nothing. **Test (36, once): geometry −3.1% (−0.134 dB [−0.204, −0.068]), simulation −6.4%, both −6.4%**; S2RNF −3.0% on its own test |
| `mesh2z/FFM_3_9_DecoderFineTune.ipynb` | Does fine-tuning the decoder on the prediction task (loss on the measured HRTF) change the geometry result? | No (development, 164): Δ = −0.016 [−0.020, −0.011] dB (FiLM row), −0.019 (biases), −0.013 (whole decoder), all below 0.046; R3 − R1 ≤ 0.006 dB; Δ is mostly the decoded-mean offset (+0.015) |
| `mesh2z/FFM_3_10_Mesh2Z_HeadRest.ipynb` | Does the head mesh outside the ear crops (REST) add to the geometry, in particular below 5 kHz? | No (development, 164): ΔLSD +0.003 dB [−0.004, +0.009], p_C1 0.95; low-band R² +0.0075 [+0.002, +0.013] measured (below the registered +0.02) against +0.089 simulated; ratio 0.084 [0.022, 0.152] |

---

## 5C. Corrections register *(additions to §5B — numbering continues)*

| # | Error | How caught | Effect on conclusions |
|---|---|---|---|
| 16 | The SYMARE ↔ CAR Lab mapping paired different people. | Asked the supervisor directly rather than accepting the spectral inference. | All 09-01…03 geometry → latent numbers withdrawn; four-database retrain; landmark study redone at N = 61. |
| 17 | The first mesh check for front/back orientation (`max x + min x > 0`) **passed** a head turned 180°, and then **failed** six correctly oriented subjects with a large occiput. | A deliberate 180° corruption of a real mesh, then the full 200-subject run. | Replaced by nose-tip height (−22.6 … −3.2 mm on all 200) plus the pinna-behind-canal check; both fail the rotated mesh. |
| 18 | The first closure check rejected 24 meshes for defects irrelevant to point sampling (all far from the ears). | Located every defective edge relative to the ear canals. | Closure is now recorded per subject and fatal only above 10 mm of defect inside the ear crop or 200 mm in total; a 12 mm hole punched at the ear canal still fails. |
| 19 | `FFM_3_1` applied the 6–12 kHz mask to the direction axis. | Failed loudly (`IndexError`, 3072 vs 93) on the first Colab run. | No wrong number produced. A shape assertion now guards the reduction. |
| 20 | `FFM_3_3` trained on principal-component scores divided by a scale factor and predicted without multiplying it back, compressing every prediction towards the mean (train R² 0.001 after 300 steps). | A capacity check before handing the notebook over: a network on 21 ears must be able to overfit, and it could not; R² also did not move when the learning rate changed tenfold. | Caught before any real run. After the fix the same check reaches train R² 0.99 — and 0.99 on shuffled targets too. |
| 21 | The 08-08 analysis (`bem_vs_measured.py` §5) reported that BEM "explains R² ≈ 0.54 of the between-subject variance" (r = 0.756). Its deviations were taken from one mean over all 122 ears, left and right together, unmirrored, so each carried the systematic left/right difference, which the BEM shares. On 10-02 that figure was set beside a per-side SONICOM figure (0.176) in a draft of this log (assistant-written) as a four-fold advantage for MRI meshes. | Harry asked where the SYMARE number came from and whether it was a BEM simulation. Recomputing both databases with one script reproduced 0.756 exactly under the original definition and gave **0.225** per side (R² as predictor **−0.524**). | The 08-08 conclusion that BEM captures individual variation is withdrawn, and so is the scan-modality lead. Both databases show r ≈ 0.2. The old script carries a correction note. |
| 22 | `FFM_3_5` (assistant-written) trained its MLP on the first 64 target PCs while ridge predicted the full latent, weighted the feature blocks differently from FFM_3_2_1, and its introduction promised five seeds where one was run. The 10-05 log entry nevertheless read the result as "the network does not extract more than a linear model". | The 10-06 fidelity audit (64 PCs keep 82% of training variance; oracle validation R² 0.63). | The MLP was below the mean either way, so the 10-05 numbers stand; the MLP-vs-ridge reading is withdrawn until FFM_3_6, which gives both the same output space and weighting. FFM_3_6 (10-06 later): with both matched, the MLP is still 0.156 dB worse than ridge (CI excludes 0), so the reading is restored on that evidence. |
| 23 | The CAR Lab exporters (`export_carlab_npy.py`, `export_carlab_44k1.py`) cached the STPS matrix under the grid size and first direction. CL126, CL154 and CL159 have 392 directions each but three different grids; CL154 and CL159 were interpolated with CL126's (rms 2.0 / 0.2 dB, max 24 / 14 dB). README known issue 6 and §8C had said the CAR Lab exporter keeps one matrix per grid. | A per-subject grid check, prompted by the supervisor's remark that the SOFA preprocessing took its direction grid from the first subject only. | The CAR Lab arrays used by FFM23/FFM26: CL154 (validation) and CL159 (training); test unaffected; reported figures not recomputed. Fixed in `preprocess_48k/` (one matrix per full-grid hash). |
| 24 | No pipeline checked each subject's left/right sense. Ear channels are exchanged in CAR Lab CL003 (lab remark "left and right mics reversed") and CL088 and in SONICOM P0339 as delivered; SONICOM P0209 is inconsistent. | Per-subject ILD/ITD check at six azimuths in `preprocess_48k.py`. | Four FFM23/FFM26 training subjects with mis-sided ears; figures not recomputed. Swapped back, and P0209 excluded, in the 48 kHz data. |
| 25 | SYMARE landmarks 31 and 32 were called "shoulder" points (§2C 09-05, the Task 1 and 1b notebooks, the first Task 1b entry). They are a left–right pair 83 mm in front of and 19 mm above the interaural midpoint, 66.9 ± 3.9 mm apart: the eyes. | Reading the coordinates while setting up Task 2 (10-11). | No number changes. The 19-landmark set has no point on the torso, which fits Task 1b (geometry predicts the head-and-ears simulation far better than the one with a torso). Notebook text corrected; the saved Task 1 run keeps the old comment. |
| 26 | §2C 09-04 recorded SuDaField as obtained and read on 09-04. Harry had read it at the end of Semester 1, sent by the supervisor, and the Unit A report already discusses it. | Harry, while writing the thesis section on the abandoned database-removal attempts (10-22). | No number changes. The early-August attempts to remove database characteristics from the latent came after SuDaField had been read; the thesis cites it as addressing the same problem, without claiming either way that it inspired them. |
| — | *Checked, not an error:* the corrected N = 61 dCor²ᵤ (0.0710) equals the invalidated N = 52 value to four decimals. | Saved outputs compared: null 95th percentile 0.0390 vs ~0.045, controls −0.0083/−0.0309 vs 0.036/0.024, p 0.0020 vs 0.0060. | A coincidence of the headline value; the run is live, not a stale cache. |
| — | *Checked, minor:* `physics_ceiling.py` truncated the 294-sample SONICOM 44.1 kHz simulations to 256 samples. | Measured on 20 subjects (10-10). | 0.07 dB rms (worst subject 0.14 dB); no conclusion changes. |
| — | *Checked, minor:* CL169's repeated directions entered the old STPS systems twice. | The repeated-direction assertion in `preprocess_48k.py`. | Averaged implicitly through the smoothing; explicit merging differs by ≤ 0.21 dB (rms 0.003 dB). |

---

## 8C. Corrections to earlier sections *(continuation of §8B)*

- **§1.3 ICASSP submission (16 September 2026).** Not submitted.
- **§9B action 1 (download meshes).** Done for SONICOM: 200 meshes plus Mesh2HRTF
  simulations held locally. HUTUBS meshes not fetched; no longer needed for mesh → latent.
- **§9B action 6 (learning curve).** Done on meshes (§2C, 10-01).
- **§9B action 10 and README known issue 6 (one STPS matrix per database).** Answered: the
  grids are identical within HUTUBS and SONICOM, so no exported array is affected.
- **§8B "no mesh file exists anywhere in the project".** Superseded by the SONICOM download.
- **SONICOM's "watertight" pre-processed meshes.** 24 of 200 are not; the graded meshes are
  what the curators simulated. Any BEM run on the pre-processed meshes must repair them first.
- **Per-dataset FiLM as a contribution (§1, §3).** Withdrawn: SuDaField published it.
  The design argument that answers the supervisor's objection — per-frequency only, 93 dof
  per database, structurally unable to absorb direction-dependent anatomy — still stands.
- **§2 (08-08) "BEM explains R² ≈ 0.54 of the between-subject variance"** and the conclusion
  built on it ("the strongest quantitative argument for the geometry-based direction") —
  withdrawn (§5C #21): per side, r = 0.225 and BEM used as a predictor scores R² = −0.52.
  **§9B action 4** is answered: the 08-08 set was `HE16kHz` (reproduced exactly with it).
- **§1.2 "geometry can't predict the notches".** Now quantified: at N = 164 dense meshes
  recover 5.6% of the 6–12 kHz gap, against 6.5% overall and a per-frequency peak near
  5 kHz.
- **§2C 09-05 "19 landmarks (13 ipsilateral pinna, 5 midsagittal, 1 shoulder)".** The last point is an eye, not a
  shoulder (§5C #25).
- **§8C "no exported array is affected" (one STPS matrix per database).** True for HUTUBS and
  SONICOM only. The CAR Lab exporters keyed their matrix cache on grid size and first direction,
  and two subjects were interpolated with another subject's grid (§5C #23).

---

## 9C. Open actions *(supersedes §9B)*

| # | Action | Why |
|---|---|---|
| 2 | Use the HUTUBS simulated/measured pairs (96 subjects) and the SONICOM Mesh2HRTF simulations (200) as Thread A validation data | Already held locally; same subjects have mesh, simulation and measurement |
| 2b | **mesh → latent is closed (10-05).** No further geometry → latent experiments; the pinna-only, LDDMM and pairwise ideas are recorded and answered or deferred to future work | — |
| 3 | *Optional:* score the frozen mesh → latent configuration once on the 36 reserved subjects | An unbiased number for the write-up; the 0.076 is a maximum over 15 configurations |
| 3a | Decide which model the 36 reserved mesh subjects (and the 15 SYMARE subjects) are spent on — the FFM_3_2_1 kernel ridge, FFM_3_6's exact ridge (the best validation row, same weighting as 3_2_1), or a learned model — before running any test cell. FFM_3_5, 3_6 and 3_7 share one test-spent marker | Each set can be used once; FFM_3_5's ridge is not the FFM_3_2_1 model (§2C, 10-05 later) |
| 4 | ~~The physics ceiling~~ — **done 10-02** (`mesh2z/physics_ceiling.py`): the scan–measurement relationship, not the regressor, is the limit | — |
| 4a | Ask the supervisor what explains the BEM–measurement gap in individual deviations (per-side r ≈ 0.2 on both SYMARE and SONICOM) | Decides whether any mesh-based route, Thread A included, can personalise against measurements |
| 4b | Agree with the supervisor what Thread A is evaluated against: BEM output (fidelity, speed) or measurements | On optical scans the BEM itself predicts measurements worse than the mean |
| 4c | Ask whether the lab holds **repeated HRTF measurements of the same ears** (beyond HUTUBS's one human pair) | Sizes the per-ear measurement term that bounds every geometry route (§2C 10-04) |
| 5 | Decide whether to mask the extrapolated lower hemisphere (14.6% of pixels, three databases) in loss and evaluation | Every reconstruction figure currently includes it |
| 6 | ~~Evaluate FFM26 on its test set, once~~ — **done 10-07**: 1.4606 dB pooled vs 4.8089 mean (§2C) | Spent |
| 8 | Decide with the supervisor whether the geometry chapter takes the positioning proposed on 10-07 (§2C: Gaps 2–4, narrowed wording); then add the references checked that day to `references.bib` and revise Ch1 contributions / Ch2 gaps | Tentative; thesis text unchanged |
| 7 | Carried over: SYMARE `Meshes/` and v1.0 BEM; full `HRIRs/AcousticRaw/`; `oslo_fast.py` integration; `FFM_2_1_latent_structure.py`; HUTUBS repeat-pair ceiling; repository stays private | Unchanged from §9B |
| 9 | ~~Train the 48 kHz autoencoder~~ — **`FFM48k_meas` done 10-10** (val 1.4793 dB); **`FFM48k_measim` done 10-11** (Task 1 encoder). `FFM48k_meas`'s test partition is read once, at the end | §2C 10-10 later, 10-11 |
| 10 | Tell the supervisor about §5C #23–24 | They touch the CAR Lab and SONICOM data FFM26 was trained on; the reversed ears are in the delivered databases themselves |
| 11 | ~~SYMARE `HTE05kHz` torso check~~ — **done 10-11** (Task 1b): the torso as scanned does not bring the simulation closer to the measurement | §2C 10-11 later |
| 12 | ~~Write and run FFM_3_9~~ — **done 10-16**: the frozen decoder is not the limit (§2C 10-16 later) | The supervisor's question whether the frozen decoder limits mesh → HRTF |
| 13 | **Thesis: write the bottleneck clearly** — what was ruled out, the bottleneck by band, the open candidates, and the wording limits, as listed in the thesis note of §2C 10-16 later | Decided 10-16; the geometry chapter's main message |
| 14 | ~~Run FFM_3_10~~ — **done 10-23**: no improvement (§2C 10-23 later). Add the row to `tab:ruled-out` | Registered 10-23 (§2C) |

---

## 7A. Use of AI tools *(addendum to §7)*

Code for `mesh2z/` (mesh sampling and checks, the notebooks FFM_3_1 to FFM_3_6, `pack_full_meshes.py`)
was written by Claude Code (Anthropic) at my request, and the assistant ran the local data audits of
09-25 and 10-01.
A second tool, Codex (OpenAI), was used on 10-06: it wrote the fidelity audit of the mesh inputs
(`notes/mesh2z_audit_2026-10-06/`: report and check scripts, outside the repository) and
`FFM_3_7_Mesh2Z_DirectFullMesh.ipynb` with `fullmesh_graph.py`. FFM_3_6 (Claude Code) was written
in response to that audit and reproduced its sampling-noise figure; FFM_3_7 shows FFM_3_6's saved
results as a reference only after confirming the same source, split and baseline.
Design choices — per-ear prediction, the dB reference levels, the identification metric —
were proposed in that discussion and accepted by me; the stopping rule was fixed before the
run (§2B), and the decision to close the arm was mine. Colab results come from runs I
executed; local audits were run on my machine. The saved outputs are the record.
The 10-07 literature check (§2C) was carried out by Claude Code from local PDFs and web sources;
statements were checked against the full text, except LAP24 and Ziegelwanger et al. 2015 (abstracts)
and Poole & Picinali 2026 (arXiv HTML).
`preprocess_48k/` and `notebooks/FFM48k_Train.ipynb` (10-10) were written by Claude Code at my request,
including the per-subject checks that found §5C #23–24 and the pre-set simulation rule; I chose the
48 kHz convention and the separate 48 kHz autoencoder, and ran the training on Colab.
`task1/Task1_SimVsMeas.ipynb` (10-11) and `task2/Task2_ShapeModes.ipynb` (10-11) were written by Claude Code at my request, its design fixed in the
notebook header before the run and gated on reproducing earlier results; I ran them on Colab. The same applies to
the `SYMARE_HTE` preprocessing, `pack_hte.py` and `task1/Task1b_TorsoCheck.ipynb` (10-11 later); the
assistant's descriptive look at all 61 SYMARE subjects before that notebook was written is recorded there.
`mesh2z/FFM_3_9_DecoderFineTune.ipynb` (10-16) was written by Claude Code at my request to test the supervisor's suggestion; its design was registered in this log before any code.
The GISNO baseline check (`analysis/gisno_baseline_check.py`, 10-27) was written and run locally by Claude Code at my request, after the authors' reply.
`notebooks/FFM48k_PipelineFigure_Arrays.ipynb` and the panel scripts for `fig:pipeline` (10-27) were written by Claude Code at my request; I ran the notebook on Colab. The diagram itself is to be composed with an image-generation tool (GPT) from these panels; every label is checked against the list in the prompt file.
`mesh2z/FFM_3_10_Mesh2Z_HeadRest.ipynb` (10-23) was written by Claude Code at my request, from a scaffold generated from FFM_3_8; I proposed the question and chose the primary statistic; its design was registered in this log before any code.
