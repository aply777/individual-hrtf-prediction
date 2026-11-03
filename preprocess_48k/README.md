# preprocess_48k — the 48 kHz data pipeline

**Created 2026-10-10.** This folder is the **48 kHz version** of the data pipeline: every database is
brought to 48 kHz and turned into HEALPix dB-magnitude arrays. It replaces, for new work,
`analysis/preprocess_multidataset_stps.py` (44.1 kHz) and the CAR Lab exporters
`analysis/export_carlab_npy.py` / `export_carlab_44k1.py`. Those scripts and their outputs, which
FFM26 was trained on, are left untouched.

Why 48 kHz: bins must line up across databases, and SYMARE and CAR Lab are native 48 kHz. HUTUBS,
the only 44.1 kHz-only database, is upsampled; SONICOM ships 48 kHz files for both its measurements
and its simulations.

## Run

```
python preprocess_48k.py            # ~30 s; writes ~/Documents/26S2/ELEC5021/data_48k/<DATASET>_48k/
python verify_48k.py                # independent checks -> verification_report.json, figures/
python pack_48k.py                  # README_DATA.md + data_48k_<date>.zip (+ .sha256)
```

`preprocess_48k.py --only HUTUBS SYMARE` processes a subset; `--overwrite` replaces existing output.
Input locations default to `~/Documents/26S1/ELEC5020` (`$ELEC5020_DATA`) and, for CAR Lab,
`analysis/_paths.py`. Needs numpy, scipy, h5py, healpy, matplotlib.

| Dataset | Input | Subjects |
|---|---|---:|
| `SONICOM` | `P*_FreeFieldCompMinPhase_48kHz.sofa` (measured) | 398 |
| `SONICOM_SIM` | `HRIR_SONICOM_48000.sofa` (Mesh2HRTF), ids paired with SONICOM | 200 |
| `HUTUBS` | `pp*_HRIRs_measured.sofa`, 44.1 kHz, upsampled | 96 |
| `SYMARE` | `HRIRs/Acoustic/HA*.mat` | 61 |
| `SYMARE_BEM` | `HRIRs/Bem/HE16kHz/HB*.mat`, ids paired with SYMARE | 61 |
| `CARLab` | `hrirfilt*`, ids from `CARLab_subjects.json` | 218 |

## Conventions (fixed before the run)

- **48 kHz everywhere.** Only HUTUBS is resampled: `resample_poly(160, 147, window=('kaiser', 10))`,
  then scaled by 44100/48000 so that the filter's frequency response is unchanged. SciPy's default
  Kaiser beta 5 left a +0.006 dB mean and 0.018 dB rms passband error (0.11 dB at the 99.9th percentile
  near deep notches); beta 10 brings these to 0.0002 / 0.0016 / 0.008 dB. Each subject is checked
  against the DTFT of its original at the same physical frequencies, up to 16 kHz.
- **Spectrum** = DTFT of the complete impulse response at f_k = k x 48000/256 (k = 0..128). Nothing is
  truncated or windowed. The SONICOM simulations are 320 samples long; cutting them to 256 would remove
  up to 21% of the energy in some directions. For a 256-sample IR this equals `rfft(h, n=256)`.
- **dB** = 20 log10(|H| + 1e-6); **STPS** order 2, lambda 1e-5, onto HEALPix nside 16 — the 44.1 kHz
  settings. `spherical_spline.py` is the single STPS implementation (identical to the copy in
  `preprocess_multidataset_stps.py` to 1.5e-14).
- **One interpolation matrix per subject grid.** The matrix is cached under a hash of the subject's
  complete direction grid.
- **Everything the processing relies on is read from each file and asserted:** sampling rate,
  `SourcePosition` type and units, receiver order (receiver 0 at +y = left), `ListenerView`.
- **Per-subject laterality check** at azimuths +-45, +-90, +-135 deg: the left ear must be louder (and,
  where the ITD is kept, earlier) for sources on the left. |ILD| must exceed 3 dB at +-90 deg and 1 dB
  elsewhere. The 1 dB bound was set after a first scan at 3 dB flagged SONICOM P0220 and HUTUBS pp33,
  whose ILD at -135 deg has the right sign but is weak (-2.0, -1.8 dB); both are normal elsewhere.
- Right ears are saved unmirrored, as before; `healpix_mirror_idx.npy` gives the mirror permutation.
- **Only up to 15.94 kHz (bins 0-85) is meant for analysis:**
  - CAR Lab has a 16.3 kHz filter.
  - The SYMARE BEM is valid to about 16 kHz.
  - HUTUBS has resampling images between 22.05 and 24 kHz.

## What the checks found in the delivered data (2026-10-10)

1. **CAR Lab grid bug in the earlier exports.**
   - `export_carlab_npy.py` and `export_carlab_44k1.py` keyed the STPS cache on grid size and first
     direction only. CL126, CL154 and CL159 all have 392 directions, but each grid is different, because
     each subject is missing a different one of the 393 directions.
   - As a result CL154 and CL159 were interpolated with CL126's grid. The error is rms 2.0 dB (max 24 dB)
     for CL154 and rms 0.2 dB (max 14 dB) for CL159, in both the 44.1 and the 48 kHz exports.
   - In FFM26's split, CL154 is in validation and CL159 in training.
2. **Ear channels exchanged.** These subjects fail the laterality check at all six azimuths, with ILD and
   ITD both reversed. They are swapped back here and flagged in the manifest. All are in FFM26's
   training partition.
   - CAR Lab CL003: the lab remark in `carlabdb.mat` reads "left and right mics reversed".
   - CAR Lab CL088: no remark.
   - SONICOM P0339: reversed in both the 44.1 and the 48 kHz files as delivered.
3. **SONICOM P0209 is excluded.**
   - Its left/right cues are inconsistent rather than swapped: ILD reversed but weak (-11/+7 dB at +-90 deg)
     and ITD erratic (+396, -375, +1458 us at +90, +135, -45 deg).
   - It is in FFM26's training partition and has no mesh, so mesh2z is unaffected.
4. **CAR Lab CL169 records six directions twice**, at azimuth +180 and -180 deg.
   - They are merged by averaging their dB magnitudes.
   - The two copies differ by 3.5 / 4.0 dB rms (left / right) over 0.2-16 kHz. This is one subject at rear
     directions, but it is a rare look at repeat-measurement variability.

None of these appears in the direction grids of the SOFA databases, which were also checked file by
file: SONICOM (measured 44.1 and 48 kHz, simulated) and HUTUBS each have exactly one grid.

## Added 2026-10-11: SYMARE head, torso and ears (`SYMARE_HTE`)

For the Task 1 torso check. The SYMARE BEM in the frozen set (`SYMARE_BEM`, `HE16kHz`) is computed on
"Head and ears" meshes with no torso; `HTE05kHz` is the same FM-BEM on "Head, torso and ears" meshes,
valid to about 5 kHz (SYMARE readme). Below 5.06 kHz the two agree to 2–3 dB rms per subject; above
5.3 kHz HTE rolls off (−13 dB at 5.8 kHz, −20 dB and lower from 6.2 kHz), so **use bins ≤ 26
(≤ 4.875 kHz) only**. Its left/right check therefore uses the ILD over 1.5–5 kHz (the 1.5–16 kHz ILD
of the rolled-off spectrum has the wrong sign at ±135° for some subjects); the ITD check is unchanged.

It is written to its own folder and zip and does not change `data_48k_2026-10-10.zip`:

```bash
python preprocess_48k.py --only SYMARE_HTE --out ~/Documents/26S2/ELEC5021/data_48k_hte
python pack_hte.py      # checks against SYMARE_BEM, writes data_48k_hte_<date>.zip + .sha256
```
