TEST 集逐人误差 —— 口径说明
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
