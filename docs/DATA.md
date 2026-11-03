# Data

None of the measured data is in this repository — see `.gitignore`. Everything below
lives on Google Drive as pre-processed `.npy` archives and is mounted by the notebooks at
run time.

> **Access.** HUTUBS, SONICOM and SYMARE are public: download them from their original
> sources and convert them with the readers described below. The CARLab archive is the CAR
> Lab's own **unpublished** measurement archive and is not distributed. The pre-processed
> archives are not linked from this repository.

---

## Archives

Every archive is **44.1 kHz**, `nfft = 256`, giving 129 frequency bins at
df = 172.27 Hz. Each `.npy` is one ear: `(129, 3072) float32`, dB magnitude, projected
onto HEALPix `nside = 16`. Training keeps bins 0-92 (DC to 15.85 kHz); see the band
discussion in `README.md`.

| Database | Subjects | Archive | Source |
|---|---:|---|---|
| **CARLab** | 218 | `CARLab_44k1_left.zip` + `CARLab_44k1_right.zip` (a folder, two files) | unpublished; not distributed |
| **SONICOM** | 399 | `SONICOM.zip` | public (SONICOM) |
| **HUTUBS** | 96 | `HUTUBS.zip` | public (HUTUBS) |
| **SYMARE (61)** | 61 | the older 61-subject set, resampled to 44.1 kHz | public (SYMARE) |

SYMARE is **not** one of the three databases the v2.3 model is trained on. It is kept
because it is the only set with 3-D meshes, which makes it the entry point for the
downstream geometry-to-latent stage. An earlier mapping of its subjects into the CARLab
archive was retired (the two cohorts are different people) and is not included.

### The sample rate is not optional

All four are at 44.1 kHz *by construction*. CARLab's delivered `hrirfilt` files are 48 kHz
and were resampled in the time domain (polyphase 147/160, `analysis/export_carlab_44k1.py`)
specifically so that a bin index means the same physical frequency in every database.
Mixing a 48 kHz archive in raises no error — it silently displaces a 12 kHz notch by about
1 kHz, which is the width of the feature the whole project is about. This has already cost
a day once; see `docs/PROJECT_LOG.md`, section 5A, correction 9.

---

## Expected layout on Drive

The notebooks resolve these paths, so keep the names:

```
MyDrive/
|- ELEC5021/CARLab/
|     CARLab_44k1_left.zip          ->  CARLab_44k1_left/CL001_left_full.npy ...
|     CARLab_44k1_right.zip         ->  CARLab_44k1_right/CL001_right_full.npy ...
|- ELEC5020/
|     HUTUBS.zip                    ->  npy_left/ , npy_right/
|     SONICOM.zip                   ->  npy_left/ , npy_right/
|- healpix_mirror_idx.npy           median-plane permutation (rebuilt with healpy if absent)
|- HRTF_Checkpoints/
      split_v21.json                the frozen train/val/test split - see below
      FFM23_best.pth                weights; also published as a GitHub Release asset
```

Subject IDs carry a database prefix: `CL001`, `HU_pp1`, `SON_P0001`. Any lookup that
strips or ignores the prefix will silently find nothing — match by suffix with a word
boundary so that `pp1` cannot match `pp19`.

### `split_v21.json`

The frozen train/val/test assignment, keyed by an md5 of each database's sorted subject
list. It exists in three places, all the same file: committed here as
`analysis/split_v21.json`, kept beside the checkpoints on Drive where the notebooks read
it, and embedded in every `.pth` under the key `split`, which is what Cell 5 of the
training notebook loads and therefore the authoritative copy. It holds subject IDs only,
no names. Do not delete or regenerate it — the CARLab partition is what makes
"single-database versus three-database on the same test list" a clean comparison.

Partition sizes: CARLab 174/22/22 subjects, HUTUBS 58/14/24, SONICOM 299/40/60.

---

## Provenance of the `.npy` archives

Every archive goes through the same three steps — FFT, `20 log10 |.|`, then a spherical
thin-plate-spline projection onto HEALPix `nside = 16`. The projection itself is
`analysis/spherical_spline.py`, a pure-Python implementation of Wahba's spherical spline.
What differs per database is only the reader in front of it.

| Database | Source format | Reader |
|---|---|---|
| CARLab | lab binary `hrirfilt` (48 kHz) and `.imp` (80 kHz raw) | `carlab_export.py` -> `export_carlab_44k1.py` -> `finalize_carlab.py`, including the time-domain 48 -> 44.1 kHz resample |
| SYMARE | MATLAB `.mat` (48 kHz) | `preprocess_multidataset_stps.py`, resampled to 44.1 kHz |
| HUTUBS | SOFA, `pp<N>_HRIRs_measured.sofa` (44.1 kHz native) | `preprocess_multidataset_stps.py` |
| SONICOM | SOFA, `P<NNNN>_FreeFieldCompMinPhase_44kHz.sofa` | `preprocess_multidataset_stps.py` |

CARLab keeps its own path because it never existed as SOFA; the other three share one
script. An earlier SYMARE converter, `mat2HEALPix.py`, is superseded — it interpolated
with `scipy.interpolate.Rbf` rather than the spherical spline, and its
`npy_full_spectrum_*` outputs were replaced by the `*_stps` ones. It is not in this
repository.

### The CARLab chain, in order

CARLab is the only database that never existed as SOFA: it arrives as the lab's binary
`hrirfilt` archive. Four scripts, run in this order:

| Step | Script | What it does |
|---:|---|---|
| 1 | `audit_carlab.py` | Inventory. `carlabdb.mat` holds 267 records, which resolve to 259 files, 241 of them readable, **218 distinct people** once duplicate names are collapsed. |
| 2 | `export_carlab_npy.py` | `hrirfilt` binary -> HEALPix `.npy` **at 48 kHz**, into a temporary directory. Azimuth is negated to match `ProcessHRIRs.m`. |
| 3 | `finalize_carlab.py` | Assigns `CL001`-`CL218` (sorted by name), splits left/right into their own folders, writes `CARLab_subjects.txt` / `.json` and `healpix_mirror_idx.npy`. |
| 4 | **`export_carlab_44k1.py`** | **This is the one training uses.** Re-exports at 44.1 kHz with a time-domain polyphase resample (147/160), reusing step 3's numbering, producing `CARLab_44k1_{left,right}` plus a `meta.json` that Cell 1 of the training notebook reads. |

Step 4 repeats step 2's work rather than resampling step 2's output, because resampling
belongs in the time domain — interpolating a magnitude spectrum smooths notches, which are
the feature the project is about. Step 2's 48 kHz export is kept because
`band_limit_audit.py` and `calib_transfer.py` are written against it.

`analysis/archive/carlab_export.py` is an early draft of step 2, from when the binary
format was still being reverse-engineered. It is kept for provenance and should not be
run.

Three supporting audits, none of which are part of the chain:

| Script | Question | Answer |
|---|---|---|
| `band_limit_audit.py` | Is CARLab's 16 kHz ceiling a measurement limit or a filter? | A filter. All 259 archives cliff at exactly 16 312.5 Hz; the raw 80 kHz `.imp` recordings hold 38-40 dB SNR from 16 to 24 kHz. |
| `head_align.py` | Does `hrirfilt` still need head-alignment correction? | No — it is already aligned (0.7-1.2 mm, +/-0.1 deg), so applying a correction changes the error by 0.00 dB. |
| `calib_transfer.py` | Is raw-versus-processed a fixed filter that could be applied to the other 163 subjects? | Mostly not. 3.81 dB total, only 0.49 dB removable by a direction-independent filter — which is why all 218 subjects go through one raw pipeline instead of mixing in the older HA files. |

### Running any of these on your own machine

The scripts used to carry an absolute path baked into a `ROOT =` line, which meant they
ran on one machine and silently nowhere else. They now resolve the data root through
`analysis/_paths.py`, in this order:

1. `$ELEC5021_ROOT`
2. the parent of this repository — `analysis/` sits at `<ROOT>/hrtf-GitHub/analysis`, so
   `<ROOT>` is two levels up, which is where `SYMARE/` lives
3. `~/Documents/26S2/ELEC5021`

If the repository is checked out inside the project folder, nothing needs setting. If it
is checked out elsewhere:

```bash
export ELEC5021_ROOT=/path/to/ELEC5021
python analysis/audit_carlab.py
```

If none of the three candidates contains a `SYMARE/` directory the scripts stop at once
and print every path they tried, rather than failing several minutes into a batch job.

### Four things to check before re-running the multi-dataset reader

None of these invalidates the archives in use; they are recorded so that the next person
to run the script — or a reviewer asking how a 440-direction grid became 3072 pixels —
finds the answer here rather than by reading the source.

1. **One STPS matrix per database, built from the first subject's grid.** Correct only
   while every subject in that database shares a direction grid. CARLab does not — it has
   393-, 475-, 412- and 392-direction subjects — which is why its own exporter caches one
   matrix per distinct grid. For HUTUBS and SONICOM this has not been asserted in code; an
   explicit check would cost one line.

2. **The dB floor differs between the two paths.** `preprocess_multidataset_stps.py` uses
   `DB_FLOOR = 1e-6` (-120 dB); the CARLab exporter uses `1e-9` (-180 dB). Neither appears
   to bind in practice — the lowest magnitude measured anywhere in the CARLab archive is
   about -80 dB, 40 dB above the shallower floor — but the two are not identical.

3. **SONICOM and HUTUBS are read from differently processed SOFA files.** SONICOM comes
   from `FreeFieldCompMinPhase`, HUTUBS from the raw measured response. Free-field
   compensation changes the magnitude spectrum. This is part of what per-dataset FiLM
   exists to absorb, and part of why a single pooled baseline across databases would be
   meaningless.

4. **Impulse responses longer than 256 samples are truncated, not windowed.** The script
   prints a warning. Minimum-phase SONICOM responses are onset-aligned and fit; measured
   HUTUBS responses should be re-checked if the script is ever run again.

The azimuth-convention question flagged in the script's own comments — whether the three
databases agree on where azimuth zero is and which way it increases — was checked
separately; see `notebooks/Check_Direction_Alignment.ipynb`.
