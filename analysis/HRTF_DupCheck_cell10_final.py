# ==========================================================================
# Cell 10 — 收尾：把 61 个全部定案，写出最终对照表
#
# 已标定的事实（Cell 8）：SYMARE61 fs=44100 / 不镜像 时，
#   51 对姓名配对中有 47 对的数据 top-1 也一致 = rank-1 92.2%（随机 0.46%）
# 所以数据方法本身可信，剩下的只是给每一行一个诚实的置信标签。
#
# 分档规则（两条独立证据链：姓名来自 carlabdb 真名，数据来自 HRTF 谱形）：
#   A  姓名与数据一致                      -> 直接用
#   B  无姓名，但数据 r 高且领先明显        -> 可用
#   C  姓名与数据冲突，但数据 r 高          -> 倾向信数据（姓名在常见姓上翻过车）
#   D  r 低 / 领先小                        -> 未解决，需查实验室记录
# ==========================================================================
import csv
import numpy as np

# 阈值：跑完 Cell 9 后按它给的"准确率>=95% 的最低阈值"填这里
R_HIGH = 0.70        # 高置信
R_LOW = 0.55         # 低于此一律不采信
M_MIN = 0.05         # 领先幅度下限

gt = {a: b for a, b in NAME_MAP.items() if a in ids_a and b in ids_b}

rows = []
for i, a in enumerate(ids_a):
    o = np.argsort(-R[i])
    cl1, r1 = ids_b[o[0]], float(R[i, o[0]])
    cl2, r2 = ids_b[o[1]], float(R[i, o[1]])
    m = r1 - r2
    nm = gt.get(a, '')
    strong = (r1 >= R_HIGH and m >= M_MIN)

    if nm and nm == cl1:
        tier, final = 'A', cl1
    elif not nm and strong:
        tier, final = 'B', cl1
    elif nm and nm != cl1 and strong:
        tier, final = 'C', cl1
    else:
        tier, final = 'D', ''

    rows.append(dict(SYMARE61=a, final_CL=final, tier=tier,
                     name_CL=nm or '-', data_CL=cl1,
                     r=round(r1, 3), runner_up=cl2, r2=round(r2, 3),
                     margin=round(m, 3),
                     rank_of_name=(int(np.where(o == ids_b.index(nm))[0][0]) + 1
                                   if nm else '')))

import collections
c = collections.Counter(r['tier'] for r in rows)
lbl = {'A': '姓名+数据一致       -> 直接用',
       'B': '无姓名，数据强      -> 可用',
       'C': '姓名冲突，数据强    -> 倾向信数据',
       'D': 'r 低/领先小        -> 未解决'}
print(f"判据: r>={R_HIGH} 且 领先>={M_MIN}\n")
for t in 'ABCD':
    print(f'  {t}  {c[t]:>2} 个   {lbl[t]}')
print(f'\n  可用 (A+B+C) = {c["A"]+c["B"]+c["C"]} / {len(rows)}')

for t in 'BCD':
    sel = [r for r in rows if r['tier'] == t]
    if not sel:
        continue
    print(f'\n--- {t} 档 ---')
    for r in sel:
        extra = (f"  姓名指 {r['name_CL']}(排第 {r['rank_of_name']})"
                 if t == 'C' else '')
        print(f"  {r['SYMARE61']:<10} -> {r['data_CL']:<7} r={r['r']:.3f} "
              f"领先={r['margin']:.3f}{extra}")

out = '/content/drive/MyDrive/HRTF_Checkpoints/symare61_to_carlab_FINAL.csv'
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'\n已写出 {out}')

# ---------------- 这才是真正的产出：哪些 CARLab 受试者有 3D mesh ----------------
have = [r for r in rows if r['final_CL']]
print(f'\n★ {len(have)} 名 CARLab 受试者可对应到 SYMARE，因而【有 3D mesh 可用】')
print('   ' + '  '.join(sorted(r['final_CL'] for r in have)))
print(f'\n   这就是 Thread B（geometry -> latent）能用的训练集规模。')
print(f'   注意上限：SYMARE 只有 61 人，所以最多 61，且是 218 人里的一个子集。')
