TEST per-subject error - conventions
(the "mean error per subject" the supervisor asked about is the rmse_band column)
================================================================

rmse_band   sqrt( mean over (frequencies in the loss band, directions) of (pred-gt)^2 )
            The SAME formula as the val figure printed each epoch during training,
            over a different scope:
              the training printout pools all elements of ALL ears into one RMSE;
              here each EAR gets its own.
            The relation is   pooled = sqrt( mean_over_ears( rmse_band^2 ) )
            -- a QUADRATIC mean, not an arithmetic one.
            So mean(rmse_band) is always <= the pooled value in _LOCKED_metrics.txt;
            the gap depends on the spread between subjects, which is exactly what
            is worth looking at.

rmse_all    as above but without the low-frequency mask. Low-frequency error is large,
            so this is clearly higher than rmse_band.
lsd         log-spectral distortion: RMS over frequency first, then an ARITHMETIC mean
            over directions. The outer mean is arithmetic, so lsd <= rmse_all.
            Use this one when comparing numbers with other papers.
mae         mean |pred-gt|, insensitive to outliers.
bias        mean (pred-gt), signed; shows systematic over- or under-estimation.
rmse_notch  RMSE computed inside 6-12 kHz (the pinna-notch band) only.
peak        largest single-point absolute error.

Note: right-ear samples are mirrored across the median plane into a left-ear frame,
so the left and right HEALPix maps can be compared directly.

(The Chinese original of this file is kept alongside as _README.zh.txt.)
