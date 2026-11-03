# --------------------- ★ FiLM β 的数据驱动初始化 ---------------------
# 各库的绝对电平差很大（实测 1-8 kHz：CARLab +3.5 / HUTUBS +2.4 / SONICOM -13.3 dB）。
# β 是 dB 域的加性偏移，正好就是增益约定 —— 与其让它从 0 学起，不如直接初始化到
# 训练集里量出来的【逐频率】库间差异。
#
# 好处：
#   · 开局 encoder 看到的三个库就已经对齐（tie_encoder_film 会做 x-β）
#   · latent 一开始就不用编码"这是哪个库的电平"（PROJECT_LOG §2 行动项 b）
#   · 不改损失定义，RMSE 口径不变，和别的 run 仍然可比
#
# γ 保持 0（增益为 1）：dB 域里整体增益就是加性偏移，β 已经覆盖；γ 表达的是
# 谱形的缩放，没有物理动机，让它自己学。
with torch.no_grad():
    prof = []
    for tr in tr_parts:
        s = torch.zeros(config['num_bins'])
        for i in range(len(tr)):
            s += tr[i][0].mean(dim=1)                 # (F,) 对方向取平均
        prof.append(s / len(tr))
    P = torch.stack(prof)                             # (D, F)
    B0 = P - P.mean(0, keepdim=True)                  # 跨库中心化
    model.decoder.film.beta.copy_(B0.to(device))

print('\nFiLM β 初始化（dB，跨库中心化后）:')
print(f"  {'库':<9}{'0.5-2k':>9}{'2-6k':>9}{'6-12k':>9}{'12-16k':>9}{'全带均值':>10}")
for k, t in enumerate(TAGS):
    r = ''.join(f'{B0[k][(freqs >= lo) & (freqs < hi)].mean():>9.2f}'
                for lo, hi in [(500, 2000), (2000, 6000), (6000, 12000), (12000, 16000)])
    print(f'  {t:<9}{r}{B0[k].mean():>10.2f}')
print('  训练后再看 Cell 5 画的 β 曲线：形状应该和这里差不多（平缓、像 EQ）。')
print('  若训练后出现 notch 形状的结构，说明 FiLM 越界去吸收人群解剖差异了。')
