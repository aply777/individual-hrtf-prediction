"""Where the data lives.

Every script in this folder that touches the CAR Lab archive imports its root from here
instead of hardcoding one. The scripts were originally written with an absolute path
baked in (`/sessions/<container-id>/mnt/ELEC5021/...`), which meant they ran on exactly
one machine, in exactly one session, and silently nowhere else.

Resolution order:

  1. `$ELEC5021_ROOT`, if set.
  2. The parent of this repository. `analysis/` sits at `<ROOT>/hrtf-GitHub/analysis`,
     so `<ROOT>` is two levels up, and that is where `SYMARE/` lives.
  3. `~/Documents/26S2/ELEC5021`, the usual location on the author's machine.

The first candidate that actually contains the directory being asked for wins. If none
does, the error names every path tried and what to do about it — a missing archive should
fail loudly at startup, not as a confusing `FileNotFoundError` several minutes into a
batch job.

    from _paths import symare_root
    ROOT = symare_root()

Scripts importing this need their own directory on `sys.path`:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _candidates():
    env = os.environ.get('ELEC5021_ROOT')
    out = []
    if env:
        out.append(os.path.abspath(os.path.expanduser(env)))
    out.append(os.path.abspath(os.path.join(_HERE, '..', '..')))
    out.append(os.path.expanduser('~/Documents/26S2/ELEC5021'))
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def project_root(require='SYMARE'):
    """Absolute path to the ELEC5021 project root (the folder holding SYMARE/)."""
    tried = _candidates()
    for c in tried:
        if require is None or os.path.isdir(os.path.join(c, require)):
            return c
    raise SystemExit(
        f"Could not locate the project root (looked for a '{require}' directory in each of):\n"
        + '\n'.join(f'  {t}' for t in tried)
        + "\n\nThe measured data is not in this repository — see docs/DATA.md. Point the\n"
          "scripts at it with:  export ELEC5021_ROOT=/path/to/ELEC5021")


def symare_root():
    """`<ROOT>/SYMARE` — the data tree: New_hrtfs/, HRIRs/, and the exported npy folders."""
    return os.path.join(project_root(), 'SYMARE')


def carlab_db():
    """The CAR Lab hrirfilt archive, as delivered by the lab."""
    return os.path.join(symare_root(), 'New_hrtfs', 'CARLabHrirDataBaseFixedByPierre')


def here(*parts):
    """A path beside this script — for figures and .npz outputs."""
    return os.path.join(_HERE, *parts)
