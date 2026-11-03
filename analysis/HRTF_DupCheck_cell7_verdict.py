# ==========================================================================
# Cell 7 — 人工判定记录（边看边填，随时重跑看进度）
#
# 看完一页就把结论填进 VERDICT。值只用三种：
#   'y'  确认是同一个人（notch 对得上，且明显优于第 2 名）
#   'n'  确认不是（第 1 名和第 2 名一样不像 / notch 全对不上）
#   '?'  看不出来（两个候选都差不多像，或数据太糊）
# 没填的默认算 '?'，不影响导出。
# ==========================================================================
import csv
import numpy as np

VERDICT = {
    # 'SYM_HA01': 'y',
    # 'SYM_HA02': 'n',
    # 'SYM_HA03': '?',
}

# ---------- 进度 ----------
done = {k: v for k, v in VERDICT.items() if v in ('y', 'n', '?')}
cnt = {v: sum(1 for x in done.values() if x == v) for v in 'yn?'}
print(f'已判定 {len(done)}/{len(ids_a)} 人   '
      f"确认 {cnt['y']} | 否定 {cnt['n']} | 存疑 {cnt['?']}")
todo = [a for a in ids_a if a not in done]
if todo:
    nxt = ids_a.index(todo[0]) // PER_PAGE
    print(f'下一个没看的是 {todo[0]}，在第 {nxt+1} 页（PAGE = {nxt}）')
else:
    print('全部看完 ✔')

# ---------- 导出 ----------
rows = []
for i, a in enumerate(ids_a):
    o = np.argsort(-R[i])
    rows.append(dict(SYMARE61=a, CARLab=ids_b[o[0]],
                     corr=round(float(R[i, o[0]]), 4),
                     runner_up=ids_b[o[1]],
                     corr2=round(float(R[i, o[1]]), 4),
                     margin=round(float(R[i, o[0]] - R[i, o[1]]), 4),
                     verdict=done.get(a, '?')))

out = '/content/drive/MyDrive/HRTF_Checkpoints/symare61_to_carlab_manual.csv'
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'\n已写出 {out}')

conf = [r for r in rows if r['verdict'] == 'y']
print(f"人工确认 {len(conf)} 对：")
print('   ' + '  '.join(f"{r['SYMARE61'][4:]}={r['CARLab']}" for r in conf))
print('\n★ 把这份和 analysis/symare61_to_carlab_final.csv（姓名 + 两次自动匹配）')
print('  交叉对照 —— 人工 + 姓名 + 数据 三方都指向同一个 CL 的，才是最可靠的。')
