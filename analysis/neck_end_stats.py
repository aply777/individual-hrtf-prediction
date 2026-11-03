"""Where the SONICOM scans end at the neck, for the captions of fig:sonicom-mesh and the Ch3 text on the head-sample cut.

python3 neck_end_stats.py <SONICOM synthetic dir> <mesh2z_samples dir>
Frame of every mesh: mm, origin at the interaural midpoint, +x forward, +y left, +z up (asserted in mesh2z/sample_meshes.py).
  - lowest point of each preprocessed scan (the input to the mesh features) and of its graded left/right meshes (Mesh2HRTF);
  - from the 50 000 area-uniform head points of mesh2z_samples: the lowest point 30 mm or more behind the interaural axis
    (x < -30, back of the neck), and the share of points below the head-sample cut z = -75 mm (FFM_3_2_1 Z_CUT).
Reads only; no model is run.
"""
import glob
import os
import sys

import numpy as np

Z_CUT, X_BACK = -75.0, -30.0


def stl_zmin(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), '<u4')[0])
        dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        return float(np.frombuffer(f.read(n * 50), dt, count=n)['v'][..., 2].min())


def summary(name, depth):
    print(f'{name:<44} min {depth.min():6.1f}  median {np.median(depth):6.1f}  max {depth.max():6.1f} mm')


def main(synth, samples):
    ids = sorted(os.path.basename(p)[:5] for p in glob.glob(os.path.join(samples, 'P*.npz')))
    pre, graded, back, below = [], [], [], []
    for s in ids:
        d = os.path.join(synth, s, 'SYNTHETIC_HRTF', s)
        pre.append(-stl_zmin(d + '_preprocessed.stl'))
        graded.append(-min(stl_zmin(d + '_graded_left.stl'), stl_zmin(d + '_graded_right.stl')))
        h = np.load(os.path.join(samples, s + '.npz'))['head_pts'].astype(float)
        back.append(-h[h[:, 0] < X_BACK, 2].min())
        below.append(100 * (h[:, 2] < Z_CUT).mean())
    pre, graded, back, below = map(np.array, (pre, graded, back, below))
    print(f'{len(ids)} subjects; depths below the interaural axis')
    summary('lowest point, preprocessed scan', pre)
    summary('lowest point, graded meshes (min of L/R)', graded)
    summary(f'lowest point at x < {X_BACK:.0f} mm (back of the neck)', back)
    print(f'scans ending above {-Z_CUT:.0f} mm at x < {X_BACK:.0f} mm: {(back < -Z_CUT).sum()} of {len(ids)}')
    print(f'head points below the cut: median {np.median(below):.1f}%, range {below.min():.1f}-{below.max():.1f}%')
    print(f'graded shallower than preprocessed by > 2 mm: {(pre - graded > 2).sum()} (max {(pre - graded).max():.1f} mm)')
    k = ids.index('P0010')
    print(f'P0010: lowest {pre[k]:.1f} mm, back {back[k]:.1f} mm, below cut {below[k]:.1f}%')


if __name__ == '__main__':
    main(*sys.argv[1:3])
