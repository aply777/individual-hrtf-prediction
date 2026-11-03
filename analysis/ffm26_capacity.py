"""FFM26 size and use of its parameters, read from the checkpoint (no training, no data).

Reports the figures quoted in the thesis paragraphs on parameter count and generalisation:
total parameters; the two dense bridge layers and everything else; the spherical convolutions;
training / validation pooled RMSE at the selected epoch (from the stored history); training
subjects and ears per database (from the stored split) and the number of spherical maps they
give (ears x frequency bins); and, for each bridge weight, how many of its singular directions
hold 99% of the squared singular values.

    python analysis/ffm26_capacity.py [path/to/FFM26_best.pth]   # -> results/ffm26_capacity.json
"""
import json
import os
import sys

import torch  # import before numpy-heavy packages (local env quirk)
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = sys.argv[1] if len(sys.argv) > 1 else f'{REPO}/results/FFM26_best.pth'
OUT = f'{REPO}/results/ffm26_capacity.json'
BRIDGE = ('encoder.fc_encode.1.weight', 'decoder.fc_decode.2.weight')

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model_state_dict']
count = lambda keys: int(sum(sd[k].numel() for k in keys))
params = [k for k, v in sd.items() if v.dtype.is_floating_point]
bridge_all = [k for k in params if k.startswith(('encoder.fc_encode.1.', 'decoder.fc_decode.2.'))]
convs = [k for k in params if k.startswith(('encoder.enc1.', 'encoder.enc2.', 'decoder.dec1.', 'decoder.dec_final.'))]  # spherical (OSLO) blocks
e = int(ck['epoch'])
h = ck['history']
n_bins = int(sd['decoder.film.beta'].shape[1])
train = {db: len(v['train']) for db, v in ck['split'].items()}
ears = 2 * sum(train.values())


def rank99(w):
    w = w.double()
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    ev = torch.linalg.eigvalsh(g).flip(0).clamp_min(0).numpy()
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1), int(len(ev))


res = dict(
    checkpoint=os.path.basename(CKPT), epoch_index=e, epoch=e + 1,
    params_total=count(params), params_bridge=count(bridge_all), params_other=count(params) - count(bridge_all),
    bridge_layers={k: dict(shape=list(sd[k].shape), weights=int(sd[k].numel()),
                           directions_for_99pct_energy=rank99(sd[k])[0], of=rank99(sd[k])[1]) for k in BRIDGE},
    params_spherical_convolutions=count(convs),
    train_rmse_db=float(h['train_rmse'][e]), val_rmse_db=float(h['val_rmse'][e]), val_rmse_db_stored=float(ck['val_rmse']),
    train_subjects=train, train_ears=ears, n_bins=n_bins, spherical_maps=ears * n_bins, values=ears * n_bins * 3072)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
print('->', OUT)
