# `mesh2z/` — plan: predicting the HRTF latent from 3-D surface meshes

**Status (2026-10-06): run.** Everything below this block is the original plan of 2026-09-03,
kept unchanged for the record; parts of it (e.g. "nothing is on disk") are out of date.
What exists now, in order:

| File | What it does |
|---|---|
| `sample_meshes.py` | 200 SONICOM meshes → 20 000-point area-uniform samples per ear crop (`mesh2z_samples`) |
| `FFM_3_1_Mesh2Z_Latents.ipynb` | per-ear latents, dB reference levels, the 164 / 36 split |
| `FFM_3_2_1_Mesh2Z_RFF.ipynb` | RFF mean embeddings + kernel ridge, nested CV (best 6.5% of the gap) |
| `FFM_3_3_Mesh2Z_PointNet.ipynb` | learned point features (worse than ridge) |
| `FFM_3_4_Mesh2Z_PinnaOnly.ipynb` | pinna-only and pinna-aligned features (worse) |
| `FFM_3_5_Geometry2Z_TrainValTest.ipynb` | fixed train/val split, epoch-trained MLPs |
| `pack_full_meshes.py`, `FFM_3_6_Mesh2Z_FullMesh.ipynb` | the complete meshes, uploaded byte for byte; features integrated exactly over every triangle instead of sampled. Validation: ridge 4.649 dB (4.3% of the gap); exact = sampled within 0.003 dB; MLP ≈ mean |
| `FFM_3_7_Mesh2Z_DirectFullMesh.ipynb`, `fullmesh_graph.py` | all original triangle coordinates plus per-vertex RFF enter a learned face encoder and shared-edge graph network; no crop, sampling, decimation or precomputed global RFF descriptor. Validation: best 4.671 dB at epoch 24 (3.7%), then overfits; below ridge |
| `physics_ceiling.py`, `sim_vs_measured.py`, `within_person.py` | why prediction is limited |

Results and their provenance are in `docs/PROJECT_LOG.md` (§2C).

---

## Why meshes, and why they are the only way to grow N

`landmark2z` showed that 33 landmark points on 52 subjects are not enough, and — more
usefully — that the failure is not one of model capacity. A network that reproduces its 33
training subjects at R² ≈ 0.99 predicts unseen people *worse than the population average*.
That is the signature of too few subjects, not of too weak a network. Richer geometry alone
would make it worse, since a mesh encoder has more parameters still.

So the next experiment has to fix **both** at once: richer geometry **and** more subjects.
Meshes are the only representation that permits this, and the reason is worth stating
precisely, because it is the same reason the HRTF side could be pooled and the landmark
side could not.

| | poolable across databases? | why |
|---|---|---|
| **HRTF direction** | **yes** | Direction is a continuous coordinate on the sphere. Any measurement grid — 393, 440, 793 points — samples the same underlying continuous function, so a thin-plate spline resamples all of them onto one HEALPix grid. That is a *resampling*, not an assumption. |
| **Landmarks** | **no** | Landmark identity is categorical and named. HUTUBS's "cavum concha height" and SYMARE's "point 3 (x, y, z)" are not two samples of one field, and no interpolation relates them. Forcing a correspondence would be an invented modelling assumption — the same class of silent error as reading a 44.1 kHz archive as 48 kHz. |
| **Surface meshes** | **yes** | A mesh is a sampling of a continuous surface. A point on the pinna is a point on the pinna regardless of which lab digitised it. Two meshes of two heads can be resampled onto a common representation — a fixed point count, a voxel grid, a spherical parameterisation — exactly as two HRTF grids were resampled onto HEALPix. |

This is the whole argument for the sub-project: **meshes restore poolability, and
poolability is what buys the subject count.**

## What that buys

| database | subjects with geometry | form | in the autoencoder? |
|---|---:|---|---|
| SONICOM | ~200 of 300 | photogrammetry, 0.5 mm point clouds, watertight, Frankfurt-plane aligned, separate graded left/right ear meshes | yes |
| HUTUBS | 58 | `.ply` head meshes (plus anthropometrics for 93) | yes |
| SYMARE | 61 | MRI-derived; HTE / HE / left ear / right ear, plus BEM-ready grades at 4–20 kHz | yes |
| **total** | **~319** | | |

All three are already inside the same autoencoder, so the **target space is shared and no
new training of the AE is needed** — the latents exist today. HUTUBS's test fraction was
deliberately set to 25% precisely because it ships meshes and its test set was reserved for
this work.

At N ≈ 300 the null distribution of the go/no-go statistic tightens from a 95th percentile
of 0.045 (at N = 52) to about 0.012. That is a different regime of statistical power, and
it is what makes a negative result at this scale worth publishing and a positive result
worth believing.

## Nothing is on disk yet

Checked 2026-09-03: `find` over the whole project returns **zero** `.ply`, `.obj` or `.stl`
files. All three mesh sets have to be obtained.

- **HUTUBS** and **SONICOM** are publicly downloadable — do these first, they are ~258 of
  the ~319 subjects and need no one's permission.
- **SYMARE** meshes must be requested. The v0.9 readme documents the whole `Meshes/` tree,
  but only `HRIRs/` was supplied to us.

## Hazards, each with a check that must fail loudly

These are ordered by how expensive they were the last time an equivalent mistake was made
in this project.

**1. Scale.** Different labs, different units — mm, m, or an arbitrary scanner unit. This
is the direct analogue of reading a 44.1 kHz archive as 48 kHz: one wrong constant, no
error message, and every downstream number quietly wrong.
*Check:* compute a physical quantity with a known human range from every mesh — interaural
distance, or head width — and assert it lands in a plausible band per database. Where
anthropometrics exist (HUTUBS ships them for 93 subjects) cross-check against the measured
value rather than against a textbook range.
**Do not normalise scale away.** Head and ear size are real anatomical signal, and the
supervisor's position is that population composition — ear and head size, age, sex — is the
signal you want when pooling, not a nuisance. Convert everything to millimetres; do not
rescale to unit size.

**2. Content.** The databases do not cover the same anatomy. SYMARE offers head+torso+ears,
head+ears, and each ear separately; SONICOM has heads, torsos and separately graded ear
meshes; HUTUBS ships head meshes. Torso inclusion in particular differs.
*Fix:* crop to a common region rather than trying to reconcile the full bodies. The
ear-centred crop is both the natural common denominator and the physically right one — the
signal `landmark2z` found sits entirely in 6–16 kHz, which is pinna-controlled. Zhou et al.
(2021) cropped and centred on the ear canal entrance and that is the precedent to follow.
*Check:* after cropping, the bounding box and point count distributions must overlap across
databases. If one database's crops are systematically larger, the crop rule is not doing
the same thing to all of them.

**3. Alignment.** SONICOM states Frankfurt-plane alignment; SYMARE's landmark export was
already in a common frame (centroid spread 1.5 mm across 61 subjects); HUTUBS is unverified.
*Fix:* define one anatomical frame and bring everything into it — Procrustes to a template
**without scaling**, or align on the ear canal entrance plus two derivable reference
directions.
*Check:* the same laterality test that settled the landmark question works here. Take the
broadband-energy-weighted mean direction of each subject's `hL` and `hR` and confirm it
points at the correct ear mesh, for every subject in every database. This caught nothing on
SYMARE only because it was checked; do not skip it because the landmark answer came out
"as expected".

**4. Left/right convention.** The same trap as the landmarks, per database. The SYMARE
readme documents `Meshes/LeftEar` and `RightEar` by name, but a directory name is not
evidence. Verify acoustically, as above.

**5. Acquisition modality.** SYMARE is **MRI**; SONICOM is optical photogrammetry at
0.5 mm; HUTUBS is unverified. These will differ systematically in how they render hair,
skin and fine ear detail — a per-database geometric bias exactly analogous to the
per-database spectral colouring that FiLM absorbs on the acoustic side, and to the 3.52 dB
direction-dependent offset between the CARLab and SYMARE processing chains.
*This is a genuine confound, not a detail.* It should be disclosed, and if a per-database
correction is ever fitted on the geometry side, note that the supervisor has already
rejected the framing of cross-database variation as bias to be removed: population
composition is the signal.

**6. Resolution.** SONICOM's 0.5 mm clouds are far denser than the BEM-ready SYMARE grades.
*Fix:* resample every mesh to a fixed point count. Zhao et al. (2024) found 2^11 = 2048
points optimal and **worse** results beyond that at N = 30; Zhou et al. (2021) found voxel
resolution between 16³ and 64³ barely mattered at N = 645. Geometric *richness* helps;
geometric *resolution* saturates early. Start at 2048 and treat anything denser as a
hypothesis to test, not an improvement to assume.

## Order of work

Cheapest first, as in `landmark2z`. Steps 1–4 fit nothing.

1. Download HUTUBS and SONICOM meshes; request SYMARE's. Run the scale, content, alignment
   and laterality checks above **before** computing a single feature.
2. Crop, align and resample to a common representation. Report the per-database
   distributions of every quantity in step 1 side by side; they must overlap.
3. Run the same go/no-go as `landmark2z` — unbiased distance correlation plus a permutation
   test — **per database, in each database's own resampled mesh features**, and then pooled.
   Per-database first: three near-zero results across three acquisition modalities is a much
   stronger statement than one pooled number, and it costs nothing extra.
4. Per band. The `landmark2z` result predicts the signal should again be confined to
   6–16 kHz. If meshes move signal into 0.5–6 kHz that is new information about what
   landmarks were missing.
5. Only then fit anything, and keep the same pre-registered stopping rule.
6. **Learning curve.** Plot the cross-validated score against subject count — 50, 100, 200,
   300. This is the single most valuable figure the larger N makes possible, because it is
   what separates "geometry does not carry this information" from "we never had enough
   people", and `landmark2z` at N = 52 could not separate them.

## Reporting discipline carried over

- Every number travels with its band, its smoothing, its reduction convention and its task.
- The reconstruction figure (1.2909 dB, HRTF in and HRTF out) never shares a table with an
  end-to-end geometry figure. They answer different questions.
- Held-out sets are spent once. The CARLab test partition is frozen and eight of its
  subjects have already been used by `landmark2z`'s end-to-end evaluation.
- A population-mean baseline accompanies every end-to-end number. It is the comparison the
  recent literature omits — Niu et al. (2025) report 4.90–5.07 dB LSD with no mean baseline
  anywhere in the paper — and it is the only one that says whether anything was learned.
