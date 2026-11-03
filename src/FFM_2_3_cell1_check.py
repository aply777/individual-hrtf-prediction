# ==========================================================================
# Cell 1 — 频率轴一致性检查
#
# 没有 meta.json 的库无法核对采样率，所以这一版改成：
#   硬检查  形状 (129, 3072) —— 这个一定要对
#   软检查  meta.json，缺了只警告
#   物证    把三个库的平均谱画在【同一条物理频率轴】上（全 129 bin，到 22.05 kHz）
#           采样率若差 8%，谱在横轴上会整体拉伸，高频边缘最明显。
#           CARLab 有一个已知地标：16.31 kHz 的数字滤波器断崖
#           （analysis/band_limit_audit.py 已证实）—— 它出现在正确位置，
#           就说明 CARLab 这条轴是对的，另外两条和它对齐就可信。
# ==========================================================================
import os, json
import numpy as np
import matplotlib.pyplot as plt

EXPECT_FS, EXPECT_NFFT, EXPECT_BINS, EXPECT_NPIX = 44100, 256, 129, 3072
DF_FULL = EXPECT_FS / EXPECT_NFFT
F_FULL = np.arange(EXPECT_BINS) * DF_FULL

print(f'假定: fs={EXPECT_FS} Hz, nfft={EXPECT_NFFT}, {EXPECT_BINS} bins, '
      f'df={DF_FULL:.3f} Hz, fmax={F_FULL[-1]:.0f} Hz')
print('=' * 86)

no_meta, curves = [], {}
for d in DATASETS:
    for side in ('left', 'right'):
        p = d[side]
        mp = os.path.join(p, 'meta.json')
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        if not m and side == 'left':
            no_meta.append(d['tag'])
        files = sorted(f for f in os.listdir(p) if f.endswith('.npy'))
        x = np.load(os.path.join(p, files[0]))
        assert x.shape == (EXPECT_BINS, EXPECT_NPIX), (
            f"{d['tag']}-{side}: 形状 {x.shape}，要求 ({EXPECT_BINS}, {EXPECT_NPIX})")
        tag_meta = (f"fs={m.get('sample_rate')} nfft={m.get('nfft')}"
                    if m else '（无 meta.json，采样率未核对）')
        print(f"  {d['tag']+'-'+side:<16}{len(files):>5} 文件 | shape {x.shape} ✔ | {tag_meta}")
        if m and (m.get('sample_rate') != EXPECT_FS or m.get('nfft') != EXPECT_NFFT):
            raise AssertionError(f"{d['tag']}-{side} 的 meta 与目标不符: {m}")

    # 取若干人算平均谱与跨方向 std（全 129 bin）
    fl = sorted(f for f in os.listdir(d['left']) if f.endswith('.npy'))[:24]
    A = np.stack([np.load(os.path.join(d['left'], f)) for f in fl])   # (n,129,3072)
    ref = A[:, (F_FULL >= 1000) & (F_FULL <= 8000), :].mean()
    curves[d['tag']] = (A.mean(axis=(0, 2)) - ref, A.std(axis=2).mean(0), ref)
    del A

print('=' * 86)
if no_meta:
    print(f'⚠️ 无 meta.json: {no_meta} —— 采样率靠下面的图人工确认')

# ---------------- 关键频段的电平（FiLM 要吸收的就是这些差异）----------------
print(f"\n各库平均电平，dB 相对自己的 1-8 kHz（横向比较看谱形，纵向差异由 FiLM 吸收）")
BANDS = [(200, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 16000),
         (16000, 18000), (18000, 22050)]
print(f"  {'库':<9}" + ''.join(f'{lo//1000}-{hi//1000}k'.rjust(9) for lo, hi in BANDS)
      + f"{'1-8k 绝对':>11}")
for t, (m_, s_, ref) in curves.items():
    row = ''.join(f'{m_[(F_FULL>=lo)&(F_FULL<hi)].mean():>9.1f}' for lo, hi in BANDS)
    print(f'  {t:<9}{row}{ref:>11.1f}')

# ---------------- 图：谱形对齐 ----------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), dpi=120)
for t, (m_, s_, _) in curves.items():
    ax[0].plot(F_FULL / 1000, m_, lw=1.8, label=t)
    ax[1].plot(F_FULL / 1000, s_, lw=1.8, label=t)
ax[0].axvline(16.3125, color='k', ls='--', lw=1.1)
ax[0].annotate('CARLab 已知断崖\n16.31 kHz', xy=(16.3125, -25), xytext=(17.5, -12),
               fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=.9))
ax[0].set_title('方向平均幅度（dB，各自以 1-8 kHz 为 0）')
ax[0].set_ylabel('dB')
ax[1].set_title('跨方向 std —— 真实 HRTF 信息在哪个频段')
ax[1].set_ylabel('dB')
for a in ax:
    a.axvspan(16, 22.05, color='gray', alpha=.10)
    a.set_xlabel('frequency (kHz)'); a.set_xlim(0, 22.05)
    a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.show()

print('怎么看这两张图：')
print('  · CARLab 的断崖必须落在虚线上。落偏了 -> CARLab 这条轴就错了。')
print('  · 另外两库的曲线若相对 CARLab 在横轴上整体拉伸/压缩，就是采样率不一致。')
print('  · 右图灰区里 std 掉到很低的库，说明它在那儿也没有真实信息 —— 印证 16 kHz 上限。')
print('  · 左图的纵向平移是电平/均衡约定的差异，这正是 FiLM 该吸收的，不用担心。')

# ---------------- 频率上限 ----------------
# 取所有库的最小上限。CARLab 在 16.31 kHz 被数字滤波器切断，所以是 16 kHz。
FREQ_CAP_HZ = 16000
DF = DF_FULL
N_BINS = int(np.floor(FREQ_CAP_HZ / DF)) + 1
print(f'\n频率上限 {FREQ_CAP_HZ} Hz -> 保留 bins 0..{N_BINS-1} ({N_BINS}/{EXPECT_BINS}), '
      f'实际 fmax {(N_BINS-1)*DF:.0f} Hz')
