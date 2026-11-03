# ==========================================================================
# Cell 0 — 数据准备
#
# 两种 zip 布局混用，靠 dirs 里的文件夹名区分，代码是同一套（递归查找）：
#   CARLab  : ELEC5021/CARLab/*_left.zip + *_right.zip  ->  CARLab_44k1_left/right
#   其余两库 : ELEC5020/<库>.zip（单个）                ->  npy_left / npy_right
#
# id 决定 FiLM 的行号，一旦训练过就不要重排。
# ==========================================================================
import os, zipfile
from google.colab import drive

drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive'
BASE = '/content/dataset'

DATASETS_SPEC = [
    dict(id=0, tag='CARLab',
         zips=[f'{DRIVE}/ELEC5021/CARLab/CARLab_44k1_left.zip',
               f'{DRIVE}/ELEC5021/CARLab/CARLab_44k1_right.zip'],
         extract_to=f'{BASE}/carlab',
         dirs=('CARLab_44k1_left', 'CARLab_44k1_right'),
         split=(0.80, 0.10, 0.10)),
    dict(id=1, tag='HUTUBS',
         zips=[f'{DRIVE}/ELEC5020/HUTUBS.zip'],
         extract_to=f'{BASE}/hutubs',
         dirs=('npy_left', 'npy_right'),
         split=(0.60, 0.15, 0.25)),          # 自带 58 个 mesh -> 多留 test
    dict(id=2, tag='SONICOM',
         zips=[f'{DRIVE}/ELEC5020/SONICOM.zip'],
         extract_to=f'{BASE}/sonicom',
         dirs=('npy_left', 'npy_right'),
         split=(0.75, 0.10, 0.15)),
]

KNOWN_SUFFIX = ['_left_full.npy', '_left.npy', '_Left.npy', '_L.npy', '-left.npy']
L2R = [('_left_full', '_right_full'), ('_left', '_right'), ('_Left', '_Right'),
       ('-left', '-right'), ('_L.', '_R.')]


def find_dir(root, target):
    """递归找名为 target 的文件夹，跳过 __MACOSX，选 .npy 最多的那个"""
    cands = []
    for dp, _, fns in os.walk(root):
        if '__MACOSX' in dp:
            continue
        if os.path.basename(dp) == target:
            cands.append((sum(f.endswith('.npy') for f in fns), dp))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


def detect_suffix(d):
    """识别 '<受试者编号><后缀>' 里的后缀。先试常见的，再退回最长公共后缀。"""
    names = sorted(f for f in os.listdir(d) if f.endswith('.npy'))
    assert names, f'{d} 里没有 .npy'
    for s in KNOWN_SUFFIX:
        if all(n.endswith(s) for n in names):
            return s, names
    s = names[0]
    for n in names[1:]:
        i = 0
        while i < min(len(s), len(n)) and s[-1 - i] == n[-1 - i]:
            i += 1
        s = s[len(s) - i:]
    return s, names


def right_of(sl):
    for a, b in L2R:
        if a in sl:
            return sl.replace(a, b)
    return None


DATASETS = []
for spec in DATASETS_SPEC:
    tag, out = spec['tag'], spec['extract_to']
    os.makedirs(out, exist_ok=True)
    ok = True
    for zp in spec['zips']:
        if os.path.exists(zp):
            with zipfile.ZipFile(zp) as z:
                z.extractall(out)
        else:
            print(f'[{tag}] ★ 找不到 {zp}'); ok = False
    if not ok:
        continue

    dl, dr = spec['dirs']
    ld, rd = find_dir(out, dl), find_dir(out, dr)
    if ld is None or rd is None:
        print(f'[{tag}] ★ 解压后没找到 {dl} / {dr}，检查 zip 结构')
        continue

    sl, ln = detect_suffix(ld)
    sr = right_of(sl)
    rn = sorted(f for f in os.listdir(rd) if f.endswith('.npy'))
    if sr is None or not all(n.endswith(sr) for n in rn):
        sr, rn = detect_suffix(rd)

    ids_l = {n[:-len(sl)] for n in ln}
    ids_r = {n[:-len(sr)] for n in rn}
    assert all(ids_l) and len(ids_l) == len(ln), \
        f'{tag}: 后缀 {sl!r} 识别有问题，切出来的编号有空或重复'
    common = sorted(ids_l & ids_r)

    print(f"[{tag}] id={spec['id']}  left {len(ln)} / right {len(rn)}  "
          f'后缀 {sl!r} / {sr!r}  双耳齐全 {len(common)} 人')
    print(f'         {ld}')
    print(f'         编号示例: {common[:4]} ... {common[-2:]}')
    if ids_l ^ ids_r:
        print(f'         ⚠️ {len(ids_l ^ ids_r)} 人只有一只耳（自动跳过）: '
              f'{sorted(ids_l ^ ids_r)[:6]}')

    DATASETS.append(dict(id=spec['id'], tag=tag, left=ld, right=rd,
                         suffix=(sl, sr), split=spec['split']))

DATASETS = sorted(DATASETS, key=lambda d: d['id'])
assert DATASETS, '一个库都没加载成功'
assert [d['id'] for d in DATASETS] == list(range(len(DATASETS))), (
    f"id 必须从 0 连续，当前是 {[d['id'] for d in DATASETS]} —— "
    '少一个库就把 DATASETS_SPEC 里后面的 id 顺次改小')

MIRROR_SRC = next((p for p in (
    f'{DRIVE}/ELEC5021/analysis/healpix_mirror_idx.npy',   # ← 现在的位置
    f'{DRIVE}/HRTF_Checkpoints/healpix_mirror_idx.npy',    # healpix_mirror.py 的旧输出
    f'{DRIVE}/healpix_mirror_idx.npy')                     # 更早的根目录版
    if os.path.exists(p)), f'{DRIVE}/ELEC5021/analysis/healpix_mirror_idx.npy')
print(f"\n共 {len(DATASETS)} 个库 | mirror idx: "
      f"{'找到' if os.path.exists(MIRROR_SRC) else '★缺失，Cell 3 会现场生成'}")

# ---- 采样率提醒：Cell 1 靠 meta.json 核对，没有就查不了 ----
no_meta = [d['tag'] for d in DATASETS
           if not os.path.exists(os.path.join(d['left'], 'meta.json'))]
if no_meta:
    print(f'\n⚠️ 这些库没有 meta.json，Cell 1 无法核对采样率: {no_meta}')
    print('   三个库必须都是 44.1 kHz / 256 点 / 129 bin。48 kHz 的 df=187.5 Hz，')
    print('   44.1 kHz 的 df=172.27 Hz —— 同一个 bin 索引差 8%，在 notch 带错位')
    print('   接近 1 kHz。混着训不会报错，只会悄悄变差。')
