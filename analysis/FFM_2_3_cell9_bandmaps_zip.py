# ==========================================================================
# Cell 10 — 逐人逐频带 HEALPix 图 + 每人平均误差 + 分卷打包下载
#
# 相对上一版的改动：
#   1. 默认切到 TEST 集（te_parts）
#   2. ★ 出图之前先跑一次标量评估并写进 _LOCKED_metrics.txt（带时间戳）
#      —— 「先记录、后观察」。顺序对了，看图就不会污染要报的数
#   3. 加了 HUTUBS 重复受试者的跨集泄漏检查（只报告，不修改任何东西）
#   4. 支持断点续跑：已完成的受试者目录会跳过，Colab 掉线不用从头来
#   5. 按数据库分卷打包，每卷控制在 180 MB 以内，浏览器能直接下
#   6. 用 dataset 自己返回的 (x, dataset_id, 'CL001_left') 三元组，
#      不再靠猜属性名拿受试者编号
#
# 关于「冻结 weight」：不需要做任何额外的事。模型里只有 GroupNorm（无 running
# statistics），没有 BatchNorm、没有 Dropout，加上 torch.no_grad()，
# 推理完全确定性，weight 一个数都不会动。下面有断言兜底。
# ==========================================================================
import os, csv, time, math, json, shutil, zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')                 # 不弹窗，批量出图快很多
import matplotlib.pyplot as plt
import healpy as hp

# ---------------- 配置 ----------------
SPLIT_NAME       = 'test'    # 'test' / 'val'
BIN_GROUP        = 5
MAX_SUBJ_PER_DB  = None      # None = 全部；先试跑可设 4
ONLY_DB          = None      # 例如 ['CARLab']；None = 全部
DPI              = 110
ZIP_PART_MB      = 180       # 单卷上限，Colab 浏览器下载超过 ~200MB 容易断
RESUME           = True      # 跳过已完成的受试者目录
OUT_ROOT         = '/content/bandmaps'
DRIVE_DIR        = '/content/drive/MyDrive/HRTF_Checkpoints'   # None = 不复制


# ==========================================================================
# 0. 频带划分（93 bin -> 19 档，最后一档 3 个 bin）
# ==========================================================================
NB    = config['num_bins']
FS    = config['sample_rate']
DF    = FS / 256.0
EDGES = [(i * BIN_GROUP, min((i + 1) * BIN_GROUP, NB))
         for i in range(int(math.ceil(NB / BIN_GROUP)))]
NBAND = len(EDGES)
F_LO  = np.array([a * DF for a, b in EDGES])
F_HI  = np.array([(b - 1) * DF for a, b in EDGES])
FREQS = np.arange(NB) * DF
BAND_MASK = FREQS >= config.get('band_lo_hz', 0)
NOTCH     = (FREQS >= 6000) & (FREQS <= 12000)

print(f'num_bins {NB} | fs {FS} | {DF:.2f} Hz/bin | 上限 {FREQS[-1]/1000:.2f} kHz')
print(f'每 {BIN_GROUP} bin 一档 -> {NBAND} 档（每档 {BIN_GROUP*DF:.0f} Hz，'
      f'末档 {EDGES[-1][1]-EDGES[-1][0]} bin）')


def band_avg(X):
    """(NB, P) -> (NBAND, P)，dB 域内平均"""
    return np.stack([X[a:b].mean(0) for a, b in EDGES])


# ==========================================================================
# 1. 选数据集
# ==========================================================================
PARTS  = {'test': 'te_parts', 'val': 'va_parts'}[SPLIT_NAME]
LOADER = {'test': 'test_loader', 'val': 'val_loader'}[SPLIT_NAME]
parts  = globals()[PARTS]
loader = globals()[LOADER]
TAG_BY_ID = {d['id']: d['tag'] for d in DATASETS}

print(f'\n使用 {SPLIT_NAME} 集（{PARTS}）')
for p in parts:
    print(f"  {TAG_BY_ID[p.dataset_id]:<10}{len(p.items)//2:>4} 人  "
          f"{len(p):>4} 只耳")


# ==========================================================================
# 2. ★ 先记录、后观察 —— 出图前把标量锁住
# ==========================================================================
stamp = time.strftime('%Y-%m-%d %H:%M:%S')
root  = f'{OUT_ROOT}_{SPLIT_NAME}_{time.strftime("%m%d_%H%M")}'
shutil.rmtree(root, ignore_errors=True) if not RESUME else None
os.makedirs(root, exist_ok=True)

# --- 冻结确认 ---
model.eval()
assert not model.training, 'model 仍在 train 模式'
n_bn = sum(1 for m in model.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)))
n_dp = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
w_sig = float(sum(p.detach().double().sum() for p in model.parameters()))

print(f'\n冻结检查: BatchNorm {n_bn} 个, Dropout {n_dp} 个, '
      f'权重校验和 {w_sig:.6f}')

# --- 标量评估（这就是进论文的数）---
print(f'\n跑 {SPLIT_NAME} 集标量评估 …')
t_ = time.time()
rmse, mae, per = evaluate(loader)
print(f'  用时 {time.time()-t_:.0f}s')

lock_lines = [
    f'{SPLIT_NAME.upper()} 集评估 —— 出图之前记录，这就是要报的数',
    f'时间 {stamp}',
    f'checkpoint {config.get("save_dir","?")}  epoch={globals().get("done_epochs","?")}',
    f'权重校验和 {w_sig:.6f}',
    '=' * 58,
    f'{"库":<10}{"人数":>6}{"耳数":>6}{"RMSE (dB)":>12}',
]
for p in parts:
    t = TAG_BY_ID[p.dataset_id]
    lock_lines.append(f'{t:<10}{len(p.items)//2:>6}{len(p):>6}'
                      f'{per.get(t, float("nan")):>12.4f}')
lock_lines += [
    f'{"pooled":<10}{"":>6}{sum(len(p) for p in parts):>6}{rmse:>12.4f}',
    f'{"MAE":<10}{"":>6}{"":>6}{mae:>12.4f}',
    '',
    '口径: RMSE over (loss band 内的频率 × 方向 × 全部耳)，与训练时每 epoch 打印的一致。',
]

# --- HUTUBS 重复受试者的跨集泄漏检查（只报告）---
lock_lines += ['', 'HUTUBS 重复受试者检查（pp1/pp96 = FABIAN 假人；pp22/pp88 = 同一人）']
try:
    sp_all = json.load(open(os.path.join(config['save_dir'], 'split_v21.json')))
    hs = sp_all['HUTUBS']
    where = lambda s: next((k for k in ('train', 'val', 'test') if s in hs[k]), '—')
    for a, b in (('pp1', 'pp96'), ('pp22', 'pp88')):
        wa, wb = where(a), where(b)
        verdict = ('★ 跨集 —— test/val 那一侧的数字受污染，报告时须说明'
                   if wa != wb and '—' not in (wa, wb) else '同侧，无泄漏')
        lock_lines.append(f'  {a:<6}in {wa:<6} | {b:<6}in {wb:<6} | {verdict}')
except Exception as e:
    lock_lines.append(f'  （未能读取 split_v21.json: {e}）')

lock_txt = '\n'.join(lock_lines)
print('\n' + lock_txt)
with open(os.path.join(root, '_LOCKED_metrics.txt'), 'w') as fh:
    fh.write(lock_txt + '\n')
if DRIVE_DIR and os.path.isdir(DRIVE_DIR):
    with open(os.path.join(DRIVE_DIR,
              f'LOCKED_{SPLIT_NAME}_{time.strftime("%m%d_%H%M")}.txt'), 'w') as fh:
        fh.write(lock_txt + '\n')
print('\n★ 以上已写入 _LOCKED_metrics.txt。把它整段贴进 PROJECT_LOG。')
print('  之后随便看图；若看完决定改模型，须在 log 写明改了什么并重跑，旧数作废。\n')


# ==========================================================================
# 3. 画图
# ==========================================================================
def draw_band(g, p, b, subj, tag, ear, path):
    e = np.abs(g - p)
    vmin, vmax = float(min(g.min(), p.min())), float(max(g.max(), p.max()))
    fig = plt.figure(figsize=(16, 4.6), dpi=DPI)
    fig.suptitle(f'{tag} / {subj} / {ear} ear'
                 f'{"  (mirrored to left-ear frame)" if ear == "right" else ""}'
                 f'   band {b:02d}   {F_LO[b]/1000:.2f}–{F_HI[b]/1000:.2f} kHz'
                 f'   MAE {e.mean():.2f} dB   max {e.max():.2f} dB',
                 fontsize=13, fontweight='bold', y=1.04)
    hp.mollview(g, fig=fig.number, sub=(1, 3, 1), title='Ground truth',
                min=vmin, max=vmax, cmap='viridis', unit='dB', format='%.1f')
    hp.mollview(p, fig=fig.number, sub=(1, 3, 2), title='Prediction',
                min=vmin, max=vmax, cmap='viridis', unit='dB', format='%.1f')
    hp.mollview(e, fig=fig.number, sub=(1, 3, 3), title='|error|',
                cmap='hot', unit='dB', format='%.1f')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_err_curve(gt, pred, subj, tag, ear, path):
    rms_f  = np.sqrt(((gt - pred) ** 2).mean(axis=1))
    bias_f = (pred - gt).mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=DPI)
    ax.plot(FREQS / 1000, rms_f, lw=2.2, color='#c0392b', label='RMS error')
    ax.plot(FREQS / 1000, bias_f, lw=1.4, color='#2471a3', ls='--',
            label='mean bias (pred−GT)')
    ax.axhline(0, color='k', lw=.6)
    ax.axvspan(6, 12, color='orange', alpha=.12, label='notch band 6–12 kHz')
    if config.get('band_lo_hz', 0) > 0:
        ax.axvspan(0, config['band_lo_hz'] / 1000, color='gray', alpha=.15,
                   label='out of loss band')
    for b in range(NBAND):
        ax.axvline(F_LO[b] / 1000, color='gray', lw=.4, alpha=.5)
    ax.set_xlabel('frequency (kHz)'); ax.set_ylabel('dB')
    ax.set_xlim(0, FREQS[-1] / 1000)
    ax.set_title(f'{tag} / {subj} / {ear} — error vs frequency   '
                 f'(overall RMS {np.sqrt(((gt-pred)**2).mean()):.3f} dB)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)
    return rms_f


def subject_errors(gt, pred):
    e = pred - gt
    return dict(
        rmse_band=float(np.sqrt((e[BAND_MASK] ** 2).mean())),   # 同训练口径
        rmse_all=float(np.sqrt((e ** 2).mean())),
        lsd=float(np.sqrt((e ** 2).mean(axis=0)).mean()),       # 文献口径
        mae=float(np.abs(e).mean()),
        bias=float(e.mean()),
        rmse_notch=float(np.sqrt((e[NOTCH] ** 2).mean())),
        peak=float(np.abs(e).max()),
    )


# ==========================================================================
# 4. 主循环（可断点续跑）
# ==========================================================================
jobs = []
for p in parts:
    tag = TAG_BY_ID[p.dataset_id]
    if ONLY_DB and tag not in ONLY_DB:
        continue
    n = len(p) if MAX_SUBJ_PER_DB is None else min(len(p), MAX_SUBJ_PER_DB * 2)
    jobs += [(p, tag, i) for i in range(n)]

print(f'待处理 {len(jobs)} 只耳 × {NBAND+1} 张 = {len(jobs)*(NBAND+1)} 张 PNG')
print(f'粗估 {len(jobs)*(NBAND+1)*0.9/60:.0f} 分钟，'
      f'{len(jobs)*(NBAND+1)*0.18:.0f} MB\n')

rows, t0, done = [], time.time(), 0
with torch.no_grad():
    for part, tag, i in jobs:
        x, did, name = part[i]                      # name 形如 'CL001_left'
        subj, ear = name.rsplit('_', 1)
        sd = os.path.join(root, tag, name)
        cache = os.path.join(sd, 'zz_errors.json')

        if RESUME and os.path.exists(cache):
            err = json.load(open(cache))
            rows.append(err); done += 1
            continue

        os.makedirs(sd, exist_ok=True)
        y = model(x.unsqueeze(0).to(device),
                  did.view(1).to(device))[0].squeeze(0).float().cpu().numpy()
        g = x.numpy()

        Gb, Pb = band_avg(g), band_avg(y)
        for b in range(NBAND):
            fn = f'band_{b:02d}_{F_LO[b]/1000:05.2f}-{F_HI[b]/1000:05.2f}kHz.png'
            draw_band(Gb[b], Pb[b], b, subj, tag, ear, os.path.join(sd, fn))
        rms_f = draw_err_curve(g, y, subj, tag, ear,
                               os.path.join(sd, 'zz_error_vs_frequency.png'))

        err = subject_errors(g, y)
        err.update(dataset=tag, subject=subj, ear=ear)
        rows.append(err); done += 1

        with open(os.path.join(sd, 'zz_summary.txt'), 'w') as fh:
            fh.write(f'{tag} / {subj} / {ear}\n' + '=' * 46 + '\n')
            for k in ('rmse_band', 'rmse_all', 'lsd', 'mae', 'bias',
                      'rmse_notch', 'peak'):
                fh.write(f'{k:<12}{err[k]:>9.4f} dB\n')
            fh.write('\nper-band RMS error (dB)\n')
            for b in range(NBAND):
                a_, b_ = EDGES[b]
                fh.write(f'  band {b:02d}  {F_LO[b]/1000:5.2f}-{F_HI[b]/1000:5.2f} kHz'
                         f'   {np.sqrt((rms_f[a_:b_]**2).mean()):.4f}\n')
        json.dump(err, open(cache, 'w'))            # ← 续跑标记

        if done % 10 == 0:
            el = time.time() - t0
            print(f'  {done}/{len(jobs)}   已用 {el/60:.1f} min   '
                  f'剩约 {el/max(done,1)*(len(jobs)-done)/60:.1f} min')

print(f'\n出图完成，用时 {(time.time()-t0)/60:.1f} 分钟')

# 权重没被动过
w_end = float(sum(p.detach().double().sum() for p in model.parameters()))
print(f'权重校验和 出图前 {w_sig:.6f} -> 出图后 {w_end:.6f}   '
      f'{"✓ 未改变" if w_sig == w_end else "★ 变了，有问题"}')


# ==========================================================================
# 5. 汇总表 + README
# ==========================================================================
FIELDS = ['dataset', 'subject', 'ear', 'rmse_band', 'rmse_all', 'lsd',
          'mae', 'bias', 'rmse_notch', 'peak']
rows.sort(key=lambda r: -r['rmse_band'])
with open(os.path.join(root, '_ALL_subject_errors.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in FIELDS})

with open(os.path.join(root, '_README.txt'), 'w') as fh:
    fh.write(f"""{SPLIT_NAME.upper()} 集逐人误差 —— 口径说明
（导师问的「每人平均误差」是 rmse_band 那一列）
================================================================

rmse_band   sqrt( mean over (loss band 内的频率, 方向) of (pred-gt)^2 )
            和训练时每 epoch 打印的 val 是【同一个公式】，只是范围不同：
              训练打印的把【全部耳】的所有元素放一起算一个 RMSE；
              这里是【一只耳】自己算一个。
            关系是   pooled = sqrt( mean_over_ears( rmse_band^2 ) )
            —— 【平方平均】，不是算术平均。
            所以 mean(rmse_band) 一定 <= _LOCKED_metrics.txt 里的 pooled，
            差多少取决于人和人之间的离散程度，这正是要看的东西。

rmse_all    同上但不做低频 mask。低频误差大，所以明显高于 rmse_band。
lsd         log-spectral distortion：先对频率求 RMS，再对方向求【算术平均】。
            外层是算术平均，所以 lsd <= rmse_all。和别人论文比数字用这个。
mae         mean |pred-gt|，对离群点不敏感。
bias        mean (pred-gt)，有符号，看系统性偏高/偏低。
rmse_notch  只在 6-12 kHz（pinna notch 带）内算的 RMSE。
peak        单点最大绝对误差。

注：右耳样本已沿中矢面镜像进「左耳坐标系」，所以左右耳的 HEALPix 图可直接对比。
""")

rb = np.array([r['rmse_band'] for r in rows])
print('\n' + '=' * 64)
print(f'耳数                        {len(rb)}')
print(f'每耳 rmse_band 算术平均      {rb.mean():.4f} dB   <- 导师要的「平均误差」')
print(f'每耳 rmse_band 平方平均      {np.sqrt((rb**2).mean()):.4f} dB   '
      f'<- 应等于上面 pooled {rmse:.4f}')
print(f'中位 / 最好 / 最差           {np.median(rb):.4f} / {rb.min():.4f} '
      f'({rows[-1]["subject"]}) / {rb.max():.4f} ({rows[0]["subject"]})')
print(f'标准差                      {rb.std():.4f} dB')
print('=' * 64)
print('\n最差的 10 只耳:')
for r in rows[:10]:
    print(f'  {r["dataset"]:<9}{r["subject"]+"_"+r["ear"]:<18}'
          f'{r["rmse_band"]:.3f} dB   notch {r["rmse_notch"]:.3f}   '
          f'bias {r["bias"]:+.3f}')


# ==========================================================================
# 6. 按数据库分卷打包
# ==========================================================================
def pack(name, dirs, extra=()):
    """把若干目录打成若干个 <= ZIP_PART_MB 的 zip，返回文件列表"""
    chunks, cur, sz = [], [], 0
    for d in dirs:
        s = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        if cur and sz + s > ZIP_PART_MB * 1e6:
            chunks.append(cur); cur, sz = [], 0
        cur.append(d); sz += s
    if cur:
        chunks.append(cur)

    out = []
    for k, ch in enumerate(chunks, 1):
        sfx = f'_part{k}of{len(chunks)}' if len(chunks) > 1 else ''
        zp = f'{root}_{name}{sfx}.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for f in extra:
                z.write(f, os.path.basename(f))
            for d in ch:
                for f in sorted(os.listdir(d)):
                    z.write(os.path.join(d, f),
                            os.path.relpath(os.path.join(d, f), root))
        out.append(zp)
        print(f'  {os.path.basename(zp):<52}{os.path.getsize(zp)/1e6:>7.1f} MB'
              f'   {len(ch)} 只耳')
    return out


extra = [os.path.join(root, f) for f in
         ('_LOCKED_metrics.txt', '_ALL_subject_errors.csv', '_README.txt')]
print('\n打包:')
zips = []
for tag in sorted({r['dataset'] for r in rows}):
    dd = sorted(os.path.join(root, tag, x)
                for x in os.listdir(os.path.join(root, tag)))
    zips += pack(tag, dd, extra)

if DRIVE_DIR and os.path.isdir(DRIVE_DIR):
    for z in zips:
        shutil.copy(z, DRIVE_DIR)
    print(f'\n全部已复制到 {DRIVE_DIR}')

print('\n浏览器下载（一次一卷，别一起点，容易断）:')
print('from google.colab import files')
for z in zips:
    print(f'files.download({z!r})')
