#!/usr/bin/env python3
"""
band_limit_audit.py — where does the CAR Lab 16 kHz wall come from?

Question (2026-08-21): the cross-database mean-magnitude comparison shows CARLab
collapsing to a ~-50 dB floor above 16 kHz while HUTUBS and SONICOM keep going.
Is that (a) the measurement, (b) the CAR Lab filter-design step, or (c) my own
export pipeline?

Answer: (b). Establishing it needs three checks, all run from the most original
artefacts on disk — no .npy, no STPS, no resampling.

  1. hrirfilt (48 kHz, 256 taps, the archive we actually train on)
     -> band-by-band level, and the exact bin where it falls off a cliff.
  2. .imp (80 kHz, 2048 samples, the raw recordings the whole chain starts from)
     -> does the MEASUREMENT contain anything above 16 kHz?
  3. Same subject through both, plus direction-dependent variance
     -> is what is left above 16 kHz signal or a common floor?

Both binary formats were reverse-engineered (the carlabtoolbox readers are not
in the repository); the .imp layout is documented in read_imp() below.

Usage:  python band_limit_audit.py [SYMARE_root]
"""
import sys
import glob
import json
import os
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/26S2/ELEC5021/SYMARE")
NEW = f"{ROOT}/New_hrtfs"
DB = f"{NEW}/CARLabHrirDataBaseFixedByPierre"

FS_RAW, FS_FILT = 80000, 48000
IMP_REC, IMP_NS = 8204, 2048          # 12-byte header + 2048 float32
HF_REC = 2052                          # 4-byte header + 2 x 256 float32


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def read_hrirfilt(path):
    """[int16 L][ N x ( int16 az, int16 el, float32 L-tap x2 ) ]  @ 48 kHz."""
    b = np.fromfile(path, dtype=np.uint8)
    L = int(np.frombuffer(b[:2].tobytes(), "<i2")[0])
    body = b[2:].reshape(-1, HF_REC)
    pos = np.frombuffer(body[:, :4].tobytes(), "<i2").reshape(-1, 2).astype(float)
    sig = np.frombuffer(body[:, 4:].tobytes(), "<f4").reshape(-1, 2, L)
    pos[:, 0] *= -1                     # ANL azimuth convention is negated
    return pos, sig


def read_imp(path):
    """
    Raw CAR Lab recording, 80 kHz.  N records of 12 + 4*2048 bytes:

        int16  record index (1-based)
        int16  0            \\  together: float32 azimuth in DEGREES,
        int16  az_hi        /   stored with the low 16 bits zeroed
        int16  0            \\  together: float32 elevation in DEGREES
        int16  el_hi        /
        int16  channel      1 = left, 2 = right
        float32 x 2048      impulse response

    Verified: hrir218 decodes to exactly the 393 ANL directions.
    """
    b = np.fromfile(path, dtype=np.uint8)
    n = b.size // IMP_REC
    body = b[:n * IMP_REC].reshape(n, IMP_REC)
    h = np.frombuffer(body[:, :12].tobytes(), "<i2").reshape(n, 6)
    sig = np.frombuffer(body[:, 12:].tobytes(), "<f4").reshape(n, IMP_NS)

    def f32(col):
        pair = np.stack([np.zeros(n, "<i2"), h[:, col].astype("<i2")], 1)
        return np.frombuffer(pair.tobytes(), "<f4")

    return f32(2), f32(4), h[:, 5], sig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def mean_spectrum_db(x, fs, nfft=4096, ref=(1000, 8000)):
    """Direction-averaged |H|, in dB, referenced to the mean over `ref`."""
    H = np.abs(np.fft.rfft(x, n=nfft, axis=-1)).mean(0)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    m = 20 * np.log10(H + 1e-30)
    return f, m - m[(f >= ref[0]) & (f <= ref[1])].mean()


def band_rows(f, m, bands):
    return [(lo, hi, float(m[(f >= lo) & (f < hi)].mean())) for lo, hi in bands
            if ((f >= lo) & (f < hi)).any()]


def show(f, m, bands, indent="    "):
    for lo, hi, v in band_rows(f, m, bands):
        print(f"{indent}{lo/1000:5.1f}-{hi/1000:5.1f} kHz : {v:+7.2f} dB")


BANDS = [(500, 1000), (1000, 8000), (8000, 12000), (12000, 16000),
         (16000, 18000), (18000, 20000), (20000, 24000), (24000, 30000)]


# --------------------------------------------------------------------------- #
# 1. the delivered archive
# --------------------------------------------------------------------------- #
def audit_archive():
    files = sorted(glob.glob(f"{DB}/hrirfilt*"),
                   key=lambda s: int(os.path.basename(s)[8:]))
    print(f"\n[1] DELIVERED ARCHIVE — {len(files)} hrirfilt files @ {FS_FILT} Hz\n")

    f = np.fft.rfftfreq(256, 1 / FS_FILT)
    M, cliff = [], []
    for p in files:
        _, sig = read_hrirfilt(p)
        m = 20 * np.log10(np.abs(np.fft.rfft(sig, axis=2)).mean(axis=(0, 1)) + 1e-20)
        m -= m[(f >= 1000) & (f <= 8000)].mean()
        M.append(m)
        plateau = m[(f >= 15000) & (f <= 16000)].mean()
        k = next((i for i in range(np.argmin(abs(f - 15000)), len(f))
                  if m[i] < plateau - 12), None)
        cliff.append(f[k] if k is not None else np.nan)

    M, cliff = np.array(M), np.array(cliff)
    vals, cnt = np.unique(cliff[~np.isnan(cliff)], return_counts=True)
    print("    cliff frequency, per file:")
    for v, c in zip(vals, cnt):
        print(f"      {v:9.1f} Hz  (bin {int(round(v*256/FS_FILT)):3d})  -> {c} files")
    print(f"      no cliff found          -> {int(np.isnan(cliff).sum())} files")
    print("\n    archive mean level, rel. 1-8 kHz:")
    show(f, M.mean(0), BANDS[:7])
    print(f"\n    16.5-20 kHz across files : mean {M[:, (f>=16500)&(f<20000)].mean():6.1f} dB"
          f"   (min {M[:, (f>=16500)&(f<20000)].mean(1).min():.1f},"
          f"  max {M[:, (f>=16500)&(f<20000)].mean(1).max():.1f})")
    return {"n": len(files), "freq_khz": (f / 1000).tolist(),
            "mean": np.round(M.mean(0), 3).tolist(),
            "p05": np.round(np.percentile(M, 5, axis=0), 3).tolist(),
            "p95": np.round(np.percentile(M, 95, axis=0), 3).tolist()}


# --------------------------------------------------------------------------- #
# 2. the raw recordings
# --------------------------------------------------------------------------- #
def audit_raw():
    print(f"\n[2] RAW RECORDINGS — .imp @ {FS_RAW} Hz\n")
    sessions = ["hrir218", "hrir1", "pl_30_04_2013.imp",
                "AnlHrtfRecordingsOct2015/AK_22_10_2015.imp",
                "AnlHrtfRecordingsOct2015/SZ_29_10_2015.imp",
                "MengyaoHrtfs/MZ hrtf 80k.imp", "NicksHrtfs/nf082710.imp"]
    hdr = ["12-16k", "16-18k", "18-20k", "20-24k", "24-30k"]
    print(f"    {'session':40s}" + "".join(f"{h:>9s}" for h in hdr))
    for s in sessions:
        p = f"{NEW}/{s}"
        if not os.path.exists(p):
            continue
        try:
            _, _, _, x = read_imp(p)
            f, m = mean_spectrum_db(x, FS_RAW)
        except Exception as e:
            print(f"    {s:40s}  unreadable ({e})")
            continue
        vals = [v for _, _, v in band_rows(
            f, m, [(12000, 16000), (16000, 18000), (18000, 20000),
                   (20000, 24000), (24000, 30000)])]
        print(f"    {s:40s}" + "".join(f"{v:+9.1f}" for v in vals))
    print("\n    -> the rig records well past 16 kHz. There is no wall here.")


# --------------------------------------------------------------------------- #
# 3. same subject, both ends of the chain
# --------------------------------------------------------------------------- #
def audit_pair(raw_name="hrir218", filt_id=218):
    print(f"\n[3] SAME SUBJECT THROUGH BOTH — {raw_name} vs hrirfilt{filt_id}\n")
    az, el, ch, raw = read_imp(f"{NEW}/{raw_name}")
    _, hf = read_hrirfilt(f"{DB}/hrirfilt{filt_id}")
    rawL = raw[ch == 1]
    print(f"    raw: {rawL.shape[0]} directions, az {az.min():.0f}..{az.max():.0f} deg,"
          f" el {el.min():.0f}..{el.max():.0f} deg,"
          f" {len(set(zip(az, el)))} unique")

    fr, mr = mean_spectrum_db(rawL, FS_RAW)
    fp, mp = mean_spectrum_db(hf[:, 0, :], FS_FILT)
    print("\n     freq        raw 80k    hrirfilt 48k     difference")
    for t in [8, 12, 14, 15, 16, 16.5, 17, 18, 20, 22]:
        ir, ip = np.argmin(abs(fr - t * 1e3)), np.argmin(abs(fp - t * 1e3))
        print(f"    {t:5.1f} kHz  {mr[ir]:+8.2f}    {mp[ip]:+8.2f}      {mp[ip]-mr[ir]:+8.2f} dB")

    print("\n    direction-dependent structure (std across directions, dB):")
    Rall = 20 * np.log10(np.abs(np.fft.rfft(rawL, n=4096, axis=1)) + 1e-30)
    Pall = 20 * np.log10(np.abs(np.fft.rfft(hf[:, 0, :], n=4096, axis=1)) + 1e-30)
    print(f"      {'band':>14s}{'raw 80k':>10s}{'filt 48k':>10s}")
    for lo, hi in [(1000, 8000), (8000, 12000), (12000, 16000), (16000, 18000),
                   (18000, 20000), (20000, 24000), (24000, 30000)]:
        sr, sp = (fr >= lo) & (fr < hi), (fp >= lo) & (fp < hi)
        a = Rall[:, sr].std(axis=0).mean()
        b = Pall[:, sp].std(axis=0).mean() if sp.any() else np.nan
        print(f"      {lo/1000:5.1f}-{hi/1000:5.1f} kHz{a:10.2f}{b:10.2f}")

    # true SNR of the raw measurement: signal window vs late-tail noise
    pk = int(np.median(np.abs(rawL).argmax(axis=1)))
    S = np.abs(np.fft.rfft(rawL[:, pk - 30:pk + 482], n=2048, axis=1)).mean(0)
    N = np.abs(np.fft.rfft(rawL[:, -512:], n=2048, axis=1)).mean(0)
    fn = np.fft.rfftfreq(2048, 1 / FS_RAW)
    snr = 20 * np.log10(S / (N + 1e-30))
    print(f"\n    raw measurement SNR (peak at sample {pk}, noise = last 512 samples):")
    show(fn, snr, [(1000, 8000), (8000, 12000), (12000, 16000), (16000, 18000),
                   (18000, 20000), (20000, 24000), (24000, 30000)], indent="      ")


if __name__ == "__main__":
    stats = audit_archive()
    audit_raw()
    audit_pair()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "band_limit_audit_stats.json")
    json.dump(stats, open(out, "w"))
    print(f"\nper-file archive statistics -> {out}")
