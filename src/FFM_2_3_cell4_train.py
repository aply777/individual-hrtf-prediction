# ==========================================================================
# Cell 4 — 训练（可随时中断，接着跑到目标轮次）
#
# 用法：TOTAL_EPOCHS 是【绝对目标】，不是"再跑多少轮"。
#   第一次跑          -> 从 1 训到 TOTAL_EPOCHS
#   第 30 轮被打断     -> 重连后按顺序跑 Cell 0~3，再跑这个 cell，
#                        它自动从第 31 轮接着跑到 TOTAL_EPOCHS
#   已经跑满还想加练   -> 把 TOTAL_EPOCHS 调大，或用 EXTEND_EPOCHS（见下）
#
# ★ 关键：学习率按【全程 TOTAL_EPOCHS 的那条 cosine】走，中断处不会重启。
#   做法是构造 scheduler 后先空转 done_epochs 步，把它推到正确位置。
#   另外必须显式写回 initial_lr —— optimizer.load_state_dict 会把 lr 恢复成
#   "中断时的值"，若不写回，scheduler 会把那个值当成 cosine 的起点，整条曲线塌掉。
#
# ★ 存盘用 临时文件 + os.replace（原子替换）。断线正好发生在写 1.2 GB 的中途时，
#   原文件不会被截断成半个损坏的存档。
# ==========================================================================
import os, time
import numpy as np
import torch

TOTAL_EPOCHS  = 150      # ← 目标总轮数（绝对）
EXTEND_EPOCHS = 50       # 已经跑满 TOTAL_EPOCHS 之后还想加练时，再跑多少轮
RESUME = 'auto'          # 'auto' = 有存档就接、没有就从零；True = 必须有；False = 强制从零

# Cell 3 里的 resume / fresh_epochs / additional_epochs 从这一版起不再使用


def save_ckpt(obj, path):
    """先写 .tmp 再原子替换，避免写到一半断线把存档写坏"""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


history = {'train_rmse': [], 'val_rmse': [], 'val_mae': [], 'train_loss_total': [],
           'per_dataset': []}
best_val_rmse, done_epochs = float('inf'), 0

ckpt = None
if RESUME and os.path.exists(latest_path):
    try:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        n_ds, n_bins = ckpt['model_state_dict']['decoder.film.beta'].shape
        if (n_bins, n_ds) != (config['num_bins'], config['num_datasets']):
            raise RuntimeError(
                f"存档是 {n_ds} 库 / {n_bins} bin，当前是 {config['num_datasets']} 库 / "
                f"{config['num_bins']} bin —— 配置变了，不能接着训")
    except Exception as e:
        if RESUME == 'auto':
            print(f'★ 存档不可用（{type(e).__name__}: {e}），改为从零开始')
            ckpt = None
        else:
            raise
elif RESUME is True:
    raise FileNotFoundError(f'RESUME=True 但没有存档: {latest_path}')

if ckpt is not None:
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    history = ckpt.get('history', history)
    done_epochs = ckpt.get('epoch', -1) + 1
    hv = history.get('val_rmse') or [ckpt.get('val_rmse', float('inf'))]
    best_val_rmse = min(hv)
    best_at = hv.index(best_val_rmse) + 1
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if done_epochs < TOTAL_EPOCHS:
        epochs, base_lr, skip = TOTAL_EPOCHS - done_epochs, config['lr'], done_epochs
        T_max, mode = TOTAL_EPOCHS, f'接着跑到第 {TOTAL_EPOCHS} 轮（原 cosine 不重启）'
    else:
        epochs, base_lr, skip = EXTEND_EPOCHS, config['lr'] * config['resume_lr_mult'], 0
        T_max = EXTEND_EPOCHS
        mode = (f'已跑满 {done_epochs} 轮，加练 {EXTEND_EPOCHS} 轮'
                f"（新 cosine，起始 lr = lr x {config['resume_lr_mult']}）")
    print(f'恢复: 已完成 {done_epochs} 轮 | 历史最佳 {best_val_rmse:.4f} dB '
          f'(第 {best_at} 轮)')
    print(f'      {mode}')
    if len(hv) >= 3:
        print(f'      最近三轮 val: ' + '  '.join(f'{v:.4f}' for v in hv[-3:]))
else:
    epochs, base_lr, skip, T_max = TOTAL_EPOCHS, config['lr'], 0, TOTAL_EPOCHS
    print(f'从零训练 {TOTAL_EPOCHS} 轮')

# ★ 必须显式写回，否则 scheduler 会拿"中断时的 lr"当 cosine 起点
for pg in optimizer.param_groups:
    pg['lr'] = pg['initial_lr'] = base_lr
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
for _ in range(skip):                      # 把 scheduler 空转到中断处
    scheduler.step()
print(f'本次要跑 {epochs} 轮 | 起始 lr {optimizer.param_groups[0]["lr"]:.3e} '
      f'| T_max {T_max}')

for ep in range(epochs):
    ep_global = done_epochs + ep + 1
    lr_used = optimizer.param_groups[0]['lr']        # 本轮真正用的 lr

    # ---------------- train ----------------
    t0 = time.time()
    model.train()
    tot, mse_sum, nb = 0.0, 0.0, 0
    for x, did, _ in train_loader:
        x, did = x.to(device), did.to(device)
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=config['amp'] and torch.cuda.is_available()):
            rec, _ = model(x, did)
            loss_mse = criterion_mse(rec[:, band_dev, :], x[:, band_dev, :])
            loss_spe = criterion_spectral(rec, x)
            loss = config['w_mse'] * loss_mse + config['w_spectral'] * loss_spe
        loss.backward(); optimizer.step()
        tot += loss.item(); mse_sum += loss_mse.item(); nb += 1
    scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train = time.time() - t0

    # ---------------- val ----------------
    t1 = time.time()
    va_rmse, va_mae, per = evaluate(val_loader)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_eval = time.time() - t1

    tr_rmse = (mse_sum / nb) ** .5
    history['train_loss_total'].append(tot / nb)
    history['train_rmse'].append(tr_rmse)
    history['val_rmse'].append(va_rmse)
    history['val_mae'].append(va_mae)
    history['per_dataset'].append(per)

    # ---------------- save ----------------
    t2 = time.time()
    is_best = False
    if ep_global > config['save_after']:
        state = {'epoch': done_epochs + ep, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'history': history,
                 'val_rmse': va_rmse, 'config': config, 'split': SPLIT,
                 'total_epochs': TOTAL_EPOCHS,
                 'datasets': [{k: v for k, v in d.items() if k != 'zips'} for d in DATASETS]}
        if ep_global % config['save_every'] == 0 or ep == epochs - 1:
            save_ckpt(state, latest_path)
        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            is_best = True
            save_ckpt(state if config['save_optimizer'] else
                      {k: v for k, v in state.items() if k != 'optimizer_state_dict'},
                      best_path)
        del state
    t_save = time.time() - t2

    # ---------------- 一行日志 ----------------
    extra = ('' if len(TAGS) == 1 else
             ' | ' + '  '.join(f'{t} {per[t]:.3f}' for t in TAGS))
    eta = (t_train + t_eval + t_save) * (epochs - ep - 1) / 60
    print(f'Epoch {ep_global:>3}/{done_epochs+epochs} | train {tr_rmse:.4f} | '
          f'val {va_rmse:.4f} dB | MAE {va_mae:.4f}' + extra
          + f' | lr {lr_used:.2e}'
          + f' | {t_train:.0f}/{t_eval:.0f}/{t_save:.0f}s'
          + f' | ETA {eta:.0f}min'
          + ('  ★best' if is_best else ''))

print(f'\n跑到第 {done_epochs + epochs} 轮。最佳 val RMSE = {best_val_rmse:.4f} dB')
if done_epochs + epochs < TOTAL_EPOCHS:
    print(f'距目标 {TOTAL_EPOCHS} 轮还差 {TOTAL_EPOCHS - done_epochs - epochs} 轮，'
          '重连后直接重跑这个 cell 即可')
for k, v in baselines.items():
    print(f'    {k} 的 mean-HRTF 基线 {v:.3f} dB')
