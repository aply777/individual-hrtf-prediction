"""All-face mesh encoder for FFM_3_7. Only NumPy and PyTorch are required.

No crop, point sampling, decimation, or precomputed global descriptor. Each STL face's
three coordinates, augmented with per-vertex RFF, enter a learned corner encoder.
CSR adjacency retains every shared-edge
connection, including non-manifold edge fans. Chunking and checkpointing retain all faces.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

GRAPH_VERSION = 'ffm37-all-faces-csr-v1'


def read_stl_bytes(b):
    if len(b) < 84:
        raise ValueError('STL shorter than its binary header')
    n = int(np.frombuffer(b, '<u4', count=1, offset=80)[0])
    if n == 0 or len(b) != 84 + 50 * n:
        raise ValueError(f'Invalid binary STL: {n} faces, {len(b)} bytes')
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    V = np.frombuffer(b, dt, count=n, offset=84)['v'].copy()
    if not np.isfinite(V).all():
        raise ValueError('Non-finite source vertex')
    return V


def face_adjacency(V):
    """CSR face graph by exact source coordinates; geometry and all face IDs are retained.

    Exact coordinate welding is used ONLY to discover adjacency. It does not replace V.
    Signed zeros denote the same point. Other nearby-but-distinct coordinates are not welded.
    Multiple faces on an edge are connected as a fan clique, without truncating their degree.
    """
    n = len(V)
    xyz = np.ascontiguousarray(V.reshape(-1, 3), dtype=np.float32).copy()
    xyz[xyz == 0] = 0.0
    key = xyz.view(np.dtype((np.void, 12))).ravel()
    _, inv = np.unique(key, return_inverse=True)
    tri = inv.reshape(n, 3)
    edge = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(n, dtype=np.int64), 3)
    order = np.lexsort((edge[:, 1], edge[:, 0]))
    edge, owner = edge[order], owner[order]
    starts = np.r_[0, np.flatnonzero(np.any(edge[1:] != edge[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(edge)]
    counts = ends - starts
    ix = starts[counts == 2]
    a, b = owner[ix], owner[ix + 1]
    sources, targets = [a, b], [b, a]
    for start, end in zip(starts[counts > 2], ends[counts > 2]):
        faces = np.unique(owner[start:end])
        aa, bb = np.meshgrid(faces, faces, indexing='ij')
        use = aa != bb
        sources.append(aa[use]); targets.append(bb[use])
    src, dst = np.concatenate(sources), np.concatenate(targets)
    keep = src != dst
    pairs = np.unique(np.stack([src[keep], dst[keep]], 1), axis=0)
    ptr = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n))].astype(np.int64)
    nbr = pairs[:, 1].astype(np.int32)
    assert len(ptr) == n + 1 and ptr[-1] == len(nbr)
    stats = dict(n_faces=n, n_exact_vertices=int(inv.max()) + 1,
                 n_directed_neighbors=len(nbr), n_boundary_edges=int(np.count_nonzero(counts == 1)),
                 n_nonmanifold_edges=int(np.count_nonzero(counts > 2)),
                 n_isolated_faces=int(np.count_nonzero(np.diff(ptr) == 0)),
                 max_degree=int(np.diff(ptr).max()), graph_version=GRAPH_VERSION)
    return ptr, nbr, stats


def prepare_graph_cache(V, source_sha256, cache_root):
    """CPU disk cache of exact triangle coordinates + CSR; never a learned-feature cache."""
    root = Path(cache_root) / (source_sha256[:20] + '_' + GRAPH_VERSION)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / 'meta.json'
    import json
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta['source_sha256'] == source_sha256 and meta['graph_version'] == GRAPH_VERSION
        vv = np.load(root / 'triangles.npy', mmap_mode='r')
        pp = np.load(root / 'ptr.npy', mmap_mode='r')
        nn_ = np.load(root / 'nbr.npy', mmap_mode='r')
        assert vv.shape == V.shape and vv.dtype == V.dtype
        assert np.array_equal(vv, V), 'Geometry cache differs from current verified STL'
        assert len(pp) == len(V) + 1 and pp[-1] == len(nn_)
        return root, meta
    ptr, nbr, stats = face_adjacency(V)
    for name, a in [('triangles', V), ('ptr', ptr), ('nbr', nbr)]:
        tmp = root / (name + '.tmp.npy')
        np.save(tmp, a); tmp.replace(root / (name + '.npy'))
    meta = dict(stats, source_sha256=source_sha256)
    tmp = root / 'meta.tmp.json'; tmp.write_text(json.dumps(meta, indent=2)); tmp.replace(meta_path)
    return root, meta


class VertexRFF(nn.Module):
    """Fixed, shared Fourier augmentation of each 3D vertex; coordinates are in mm.

    Wavevectors are angular spatial frequencies in rad/mm: phase = p @ K.T,
    with NO extra 2*pi. All subjects/corners share K, saved as a state_dict buffer.
    This layer does not reduce the face/corner axes or replace the raw coordinates.
    """
    def __init__(self, features_per_scale=16, sigmas_rad_per_mm=(.02, .1, .5, 2.), seed=37):
        super().__init__()
        m = int(features_per_scale)
        sigmas = np.asarray(sigmas_rad_per_mm, dtype=np.float64)
        if m < 0 or sigmas.ndim != 1 or not len(sigmas) or not np.isfinite(sigmas).all() or (sigmas <= 0).any():
            raise ValueError('RFF needs non-negative feature count and finite positive scales')
        rng = np.random.default_rng(int(seed))
        K = np.concatenate([rng.standard_normal((m, 3)) * s for s in sigmas], axis=0).astype(np.float32)
        self.register_buffer('wavevectors_rad_per_mm', torch.from_numpy(K))
        self.feature_dim = 2 * len(K)
        self.amplitude = 1. / math.sqrt(len(K)) if len(K) else 1.

    def forward(self, positions_mm):
        if not self.feature_dim:
            return positions_mm.new_empty((*positions_mm.shape[:-1], 0))
        phase = positions_mm @ self.wavevectors_rad_per_mm.T
        # Keep total Fourier feature energy at one as the frequency count changes.
        return torch.cat([torch.cos(phase), torch.sin(phase)], -1) * self.amplitude


class CornerEncoder(nn.Module):
    def __init__(self, width, use_rff=True, rff_features_per_scale=16,
                 rff_sigmas_rad_per_mm=(.02, .1, .5, 2.), rff_seed=37):
        super().__init__()
        self.rff = VertexRFF(rff_features_per_scale if use_rff else 0, rff_sigmas_rad_per_mm, rff_seed)
        # Existing 13 geometric inputs + explicit raw xyz (3) + per-vertex RFF (128 by default).
        self.corner = nn.Sequential(nn.Linear(16 + self.rff.feature_dim, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.face = nn.Sequential(nn.Linear(74, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, V, canal, flip):
        # One ear frame for ALL source faces. Right reflection reverses face winding as well.
        Q = V * flip
        center = Q.mean(1)
        cross = torch.linalg.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]) * flip.prod()
        area2 = cross.norm(dim=1)
        normal = cross / area2.clamp_min(1e-20)[:, None]
        rel = center - canal * flip
        geom = torch.cat([center / 100., rel / 100., normal, torch.log1p(area2 / 2)[:, None]], 1)
        # Fixed physical divisors, NOT per-person normalization. Every original corner enters.
        offsets = (Q - center[:, None]) / 5.
        positions_mm = Q - canal * flip
        x = torch.cat([geom[:, None].expand(-1, 3, -1), offsets, Q / 100., self.rff(positions_mm)], -1)
        h = self.corner(x)
        return self.face(torch.cat([h.mean(1), h.amax(1), geom], 1))


class FaceGraphBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, neighbors, offsets, ids):
        # Empty bags (boundary/isolated faces) produce zero; no faces are dropped.
        near = F.embedding_bag(neighbors, h, offsets, mode='mean', include_last_offset=True)
        own = h.index_select(0, ids)
        return self.norm(own + self.mix(torch.cat([own, near, near - own], 1)))


class FullMeshGraphRegressor(nn.Module):
    def __init__(self, out_dim, width=64, n_blocks=3, head_width=128, dropout=.1,
                 chunk_faces=32768, checkpoint_blocks=True, soft_radii_mm=(15., 40., 100.),
                 use_rff=True, rff_features_per_scale=16, rff_sigmas_rad_per_mm=(.02, .1, .5, 2.), rff_seed=37):
        super().__init__()
        self.encoder = CornerEncoder(width, use_rff, rff_features_per_scale, rff_sigmas_rad_per_mm, rff_seed)
        self.blocks = nn.ModuleList([FaceGraphBlock(width) for _ in range(n_blocks)])
        self.chunk_faces = int(chunk_faces)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.soft_radii_mm = tuple(map(float, soft_radii_mm))
        d = (2 + len(self.soft_radii_mm)) * width + 7
        self.head = nn.Sequential(nn.Linear(d, head_width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_width, out_dim))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        self.last_n_faces = None

    def _run(self, fn, *args):
        if self.training and torch.is_grad_enabled() and self.checkpoint_blocks:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    def forward(self, V, ptr_cpu, neighbors, canal, side=0):
        n = len(V)
        if n == 0 or len(ptr_cpu) != n + 1:
            raise ValueError('Full face graph shape mismatch')
        flip = V.new_tensor([1., -1. if int(side) else 1., 1.])
        chunks = [(a, min(a + self.chunk_faces, n)) for a in range(0, n, self.chunk_faces)]
        self.last_n_faces = sum(b - a for a, b in chunks)
        assert self.last_n_faces == n, 'A tail chunk was omitted'
        h = torch.cat([self._run(self.encoder, V[a:b], canal, flip) for a, b in chunks], 0)
        for block in self.blocks:
            out = []
            for a, b in chunks:
                lo, hi = int(ptr_cpu[a]), int(ptr_cpu[b])
                offsets = torch.as_tensor(ptr_cpu[a:b + 1] - lo, dtype=torch.long, device=V.device)
                ids = torch.arange(a, b, device=V.device)
                out.append(self._run(block, h, neighbors[lo:hi], offsets, ids))
            h = torch.cat(out, 0)
        # These reductions are INSIDE the learned network, after every face has been encoded.
        Q = V * flip
        center = Q.mean(1)
        area = .5 * torch.linalg.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]).norm(dim=1)
        total = area.sum().clamp_min(1e-20)
        pools = [(h * (area / total)[:, None]).sum(0), h.amax(0)]
        dist2 = ((center - canal * flip) ** 2).sum(1)
        for radius in self.soft_radii_mm:
            w = area * torch.exp(-.5 * dist2 / radius ** 2)
            pools.append((h * (w / w.sum().clamp_min(1e-20))[:, None]).sum(0))
        # Both global reductions include the entire head: soft ear readouts are not hard crops.
        bounds = Q.reshape(-1, 3).amax(0) - Q.reshape(-1, 3).amin(0)
        physical = torch.cat([canal * flip / 100., bounds / 100., torch.log1p(total)[None] / 10.])
        return self.head(torch.cat([*pools, physical]))


def to_device_mesh(V, ptr, nbr, device):
    # Copies keep read-only memory mapped arrays out of torch's writable storage contract.
    return (torch.from_numpy(np.array(V, copy=True)).to(device), np.array(ptr, copy=True),
            torch.from_numpy(np.array(nbr, dtype=np.int64, copy=True)).to(device))
