# HPC_GUIDE — Running clax + clax-pt on GPU Clusters

Operational guide for submitting clax and clax-pt benchmarks and validation
jobs on two SLURM clusters: **Raven** (MPCDF, A100) and **igpu** (Kavli IPMU, V100).

---

## Table of contents

1. [Cluster quick-reference](#1-cluster-quick-reference)
2. [Raven (MPCDF A100)](#2-raven-mpcdf-a100)
3. [igpu (Kavli IPMU V100)](#3-igpu-kavli-ipmu-v100)
4. [Available sbatch scripts](#4-available-sbatch-scripts)
5. [What each benchmark script produces](#5-what-each-benchmark-script-produces)
6. [CLASS-PT setup (required for EPT/accuracy scripts)](#6-class-pt-setup)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Cluster quick-reference

| | Raven (MPCDF) | igpu (Kavli IPMU) |
|---|---|---|
| GPU | NVIDIA A100-SXM4-80GB | Tesla V100-SXM2-32GB |
| SLURM partition | `gpu` (via `--constraint="gpu"`) | `main` (only partition) |
| GPU gres | `--gres=gpu:a100:1` | `--gres=gpu:1` |
| CPUs per task | 18 | 6 |
| Memory | 125 GB | 125 GB |
| Time limit | 24 h max | infinite (use 7-00:00:00 as ceiling) |
| Python env | conda — `py311forge` (Python 3.11) | micromamba — `clax` (Python 3.14) |
| Module system | yes — `module load` | none — micromamba directly |
| CUDA | 12.2 | 12.9 (driver 575.57.08) |
| JAX | jax-cuda12 pip wheel in `py311forge` | 0.9.2 pip wheel in `clax` env |
| Home | `/u/minh/` | `/home/n2minh/` |
| Dev repo | `/u/minh/clax` | `/home/n2minh/clax` |
| **Submit from** | `/u/minh/clax` (repo) | `/lustre/work/n2minh/clax` (lustre clone) |
| Benchmark output | `/ptmp/minh/clax-bench/<DATE>/` | `/lustre/work/n2minh/clax/benchmark/<DATE>/` |
| SLURM stdout/err | `/ptmp/minh/std/clax/benchmark/` | `/lustre/work/n2minh/std/clax/benchmark/` |
| CLASS-PT matrices | `$HOME/CLASS-PT/pt_matrices/` | `/home/n2minh/CLASS-PT/pt_matrices/` |

---

## 2. Raven (MPCDF A100)

### 2.1 Node specs

- **GPU:** NVIDIA A100-SXM4-80GB (80 GB HBM2e)
- **CPUs:** 72 physical cores/node — 18 allocated per GPU job
- **Memory:** up to 512 GB/node; 125 GB per job
- **Interconnect:** InfiniBand HDR

### 2.2 Filesystems

| Path | Purpose | Notes |
|---|---|---|
| `/u/minh/` | Home — permanent | Quota-limited; store code here |
| `/ptmp/minh/` | Scratch — fast parallel FS | **Auto-purged after ~14–30 days** — use for benchmark output only |
| `/ptmp/minh/std/clax/benchmark/` | SLURM stdout/err | Create once with `mkdir -p` |
| `/ptmp/minh/clax-bench/` | Benchmark results | Created by sbatch scripts automatically |

### 2.3 Python environment

Raven uses the module system. The clax environment is a pre-built conda env at
a fixed path — do not recreate it, just activate it.

```bash
module purge
module load anaconda/3/2023.03   # latest Anaconda on Raven
module load cuda/12.2            # CUDA 12.x, compatible with jax-cuda12 wheels
conda activate /u/minh/conda-envs/py311forge
```

Key packages in `py311forge`:
- Python 3.11, NumPy, SciPy, matplotlib
- JAX + jaxlib with CUDA 12 support (pip wheel)
- diffrax, equinox, jaxtyping
- clax + clax.ept (editable install from `/u/minh/clax`)

**LD_LIBRARY_PATH — required.** jax[cuda12] pip wheels install NVIDIA
libraries under `site-packages/nvidia/*/lib/`, which is not on
`LD_LIBRARY_PATH` by default. All sbatch scripts handle this:

```bash
NVIDIA_SP=/u/minh/conda-envs/py311forge/lib/python3.11/site-packages/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
```

JAX silently falls back to CPU if this block is missing.

### 2.4 Submission workflow

```bash
cd /u/minh/clax
git pull origin <branch>

# Fast suite (~30–60 min, excluding planck_cl)
sbatch slurm/bench-a100-raven-fast.sbatch

# planck_cl only (up to 24 h) — can run simultaneously on a second node
sbatch slurm/bench-a100-raven-planck_cl.sbatch

# Full suite: fast + planck_cl in a single job (up to 24 h)
sbatch slurm/bench-a100-raven.sbatch
```

Monitor:
```bash
squeue -u minh
tail -f /ptmp/minh/std/clax/benchmark/clax-bench-fast.out.<JOBID>
ls -lt /ptmp/minh/clax-bench/<DATE>/
```

### 2.5 SLURM header reference (Raven)

```bash
#!/bin/bash -l              # -l required: makes module available
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000        # MB
#SBATCH --time=02:00:00     # 24:00:00 for full/planck_cl jobs
#SBATCH -o /ptmp/minh/std/clax/benchmark/%x.out.%j
#SBATCH -e /ptmp/minh/std/clax/benchmark/%x.err.%j
#SBATCH -D ./               # working directory = submission directory
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
```

`--constraint="gpu"` selects the GPU partition; there is no separate
`--partition` flag needed on Raven.

---

## 3. igpu (Kavli IPMU V100)

### 3.1 Node specs

- **GPU:** Tesla V100-SXM2-32GB (32 GB HBM2) — 4 GPUs per node
- **CPUs:** 6 allocated per job
- **Memory:** 125 GB per job
- **Nodes:** `igpu01`, `igpu02`, `igpu04`–`igpu08`
  (`igpu03` is currently **drained** — SLURM routes around it automatically
  when `--nodelist` includes all eight names)
- **Time limit:** `infinite` (no hard cap enforced) — use `--time=7-00:00:00`
  as a safety ceiling

### 3.2 Filesystems

| Path | Purpose | Notes |
|---|---|---|
| `/home/n2minh/` | Home — permanent | Store code and CLASS-PT matrices here |
| `/home/n2minh/clax` | Dev repo — **edit here** | Never submit jobs from here |
| `/lustre/work/n2minh/clax` | Lustre clone — **submit from here** | Always `git pull` before submitting |
| `/lustre/work/n2minh/clax/benchmark/<DATE>/` | Benchmark output | Created by sbatch scripts |
| `/lustre/work/n2minh/std/clax/benchmark/` | SLURM stdout/err | Create once with `mkdir -p` |

Always submit from the lustre clone — it is visible from compute nodes.

### 3.3 Python environment

igpu has **no module system**. Python is managed entirely with micromamba.

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
```

The `clax` micromamba env:
- Python 3.14
- JAX 0.9.2 + jaxlib with CUDA 12.9 support (pip wheel)
- NumPy 2.x — **`np.trapz` is removed**; use `np.trapezoid`
- diffrax, equinox, jaxtyping
- clax + clax.ept (editable install from `/lustre/work/n2minh/clax`)

CUDA 12.9 (driver 575.57.08) — JAX detects the GPU automatically once
`LD_LIBRARY_PATH` is set (all sbatch scripts handle this):

```bash
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
```

### 3.4 Submission workflow

```bash
# 1. Edit and commit in the dev repo
cd /home/n2minh/clax
git add <files> && git commit -m "..." && git push origin <branch>

# 2. Pull into the lustre clone
cd /lustre/work/n2minh/clax
git pull origin <branch>

# 3. Submit
sbatch slurm/bench-v100-fast.sbatch          # fast suite (~4 h)
sbatch slurm/bench-v100-planck_cl.sbatch     # planck_cl only (~48 h)
sbatch slurm/bench-v100-igpu.sbatch          # pytest + notebook + full suite (up to 7 d)
```

The fast and planck_cl jobs can run simultaneously on different nodes.

Monitor:
```bash
squeue -u n2minh
ls -lt /lustre/work/n2minh/clax/benchmark/<DATE>/
tail -f /lustre/work/n2minh/std/clax/benchmark/clax-bench-fast.out.<JOBID>
```

**`InvalidAccount` on submission:** a transient SLURM artefact that appears
briefly after submission and resolves within ~30 s. No action needed.

### 3.5 SLURM header reference (igpu)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=7-00:00:00    # 4:00:00 for fast; 48:00:00 for planck_cl
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
```

No `--partition` flag — `main` is the only partition.

---

## 4. Available sbatch scripts

All scripts live in `slurm/` in the repo root.

| Script | Cluster | What it runs | Wall time |
|---|---|---|---|
| `bench-a100-raven-fast.sbatch` | Raven | solvers, gradients, fit_cl, ept, clpp, accuracy | 2 h |
| `bench-a100-raven-planck_cl.sbatch` | Raven | planck_cl only | 24 h |
| `bench-a100-raven.sbatch` | Raven | full suite (fast + planck_cl in one job) | 24 h |
| `bench-v100-fast.sbatch` | igpu | solvers, gradients, fit_cl, ept, clpp, accuracy | 4 h |
| `bench-v100-planck_cl.sbatch` | igpu | planck_cl only | 48 h |
| `bench-v100-igpu.sbatch` | igpu | pytest --fast + full validation notebook + all timing scripts | 7 d |

**Recommended pair for a full benchmark run:**

```bash
# Raven — two jobs, can run simultaneously
sbatch slurm/bench-a100-raven-fast.sbatch
sbatch slurm/bench-a100-raven-planck_cl.sbatch

# igpu — two jobs, can run simultaneously
sbatch slurm/bench-v100-fast.sbatch
sbatch slurm/bench-v100-planck_cl.sbatch
```

Use `bench-v100-igpu.sbatch` when you also need pytest validation
(e.g., after a code change that may have broken tests).

---

## 5. What each benchmark script produces

Scripts write one `.txt` file each. Output lands at
`benchmark/<DATE>/<node>-<gpu>-<suffix>.txt`.

| Python script | File suffix | Key quantities reported |
|---|---|---|
| `benchmark_solvers.py` | `-solvers.txt` | Per-solver median wall time, speedup vs kvaerno5 |
| `benchmark_gradients.py` | `-gradients.txt` | AD vs FD for `P(k)` derivatives; \|AD/FD−1\| per cosmological param |
| `benchmark_speed.py fit_cl` | `-fit_cl.txt` | Compile + cached time; accuracy at ℓ = 100, 500 |
| `benchmark_speed.py planck_cl` | `-planck_cl.txt` | Compile + cached time; accuracy at ℓ = 20, 100, 1000 |
| `benchmark_ept.py` | `-ept.txt` | EPT forward time per z, multi-z total, AD gradient d(ΣP)/d(ω_b) |
| `benchmark_clpp.py` | `-clpp.txt` | C_ℓ^pp timing for none/halofit/ept; BB mode ratios |
| `accuracy_classpt.py` | `-accuracy.txt` | Max/mean relative error per spectrum vs CLASS-PT (k < 0.3 h/Mpc) |

`bench-v100-igpu.sbatch` additionally produces:
- `-pytest_fast.txt` — pytest --fast summary (pass/fail counts)
- `-notebook.txt` — nbconvert execution log for the full validation notebook
- `REPORT.md` — aggregated summary of all outputs for the run

---

## 6. CLASS-PT setup

The `accuracy_classpt.py`, `benchmark_ept.py`, and `benchmark_clpp.py`
scripts load one-loop PT matrices from CLASS-PT. These must exist before
submitting any job that calls those scripts.

### Required location

| Cluster | Path | Files |
|---|---|---|
| igpu | `/home/n2minh/CLASS-PT/pt_matrices/` | 40 `.dat` files |
| Raven | `$HOME/CLASS-PT/pt_matrices/` | 40 `.dat` files |

### One-time setup

```bash
# Clone CLASS-PT into home (not into /ptmp or /lustre — they may be purged)
git clone https://github.com/<CLASS-PT-repo>/CLASS-PT.git $HOME/CLASS-PT

# Verify the matrix files
ls $HOME/CLASS-PT/pt_matrices/ | wc -l      # should be 40
ls $HOME/CLASS-PT/pt_matrices/M22oneline_N256.dat   # spot-check one file
```

If matrices are missing the scripts fail immediately with:
```
FileNotFoundError: .../pt_matrices/M22oneline_N256.dat
```

---

## 7. Troubleshooting

### JAX sees no GPU

```
No GPU/TPU found, falling back to CPU.
```

The `LD_LIBRARY_PATH` block is absent or the NVIDIA site-packages path is
wrong. Verify:

```bash
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]

# On igpu — check the site-packages path
python -c "import site; print(site.getsitepackages()[0])"
# Append /nvidia/*/lib and check .so files are present
```

---

### `AttributeError: module 'numpy' has no attribute 'trapz'`

NumPy 2.0 removed `np.trapz`. The igpu `clax` env uses NumPy 2.x.
Replace all occurrences with `np.trapezoid` (identical call signature).
Fixed in commit `219d0ff` on `benchmark/clax-pt`.

---

### `ConcretizationTypeError: float() on a traced value`

A `float()` call on a JAX-traced value inside a `jax.grad` / `jax.jit`
trace. The fix pattern is:

```python
# Wrong — raises ConcretizationTypeError under jax.grad
h = float(params.h)

# Correct — keep traced for AD; use stop_gradient only for numpy-bound ops
h = params.h                                   # JAX scalar, stays traced
h_conc = float(jax.lax.stop_gradient(h))      # concrete copy for numpy
```

For the IR resummation path in `compute_ept_from_clax`, pre-compute with
`stop_gradient` and pass as `_ir_precomputed` to activate the
gradient-safe code path. Fixed in commit `8a2438b` on `benchmark/clax-pt`.

---

### pytest hangs for hours on igpu

`--fast` limits test count but some cold JIT compilations still take
10–30 min. If `pytest_fast.txt` shows no new output for more than 3 h,
the job is stuck:

```bash
scancel <JOBID>
cd /lustre/work/n2minh/clax && git pull origin <branch>
sbatch slurm/bench-v100-igpu.sbatch
```

---

### `InvalidAccount` immediately after `sbatch` on igpu

Transient SLURM state — resolves within ~30 s. Check again with
`squeue -u n2minh`; the job will be in `PD` or `R` state shortly.

---

### Raven: `module: command not found`

The sbatch script must use a login shell: `#!/bin/bash -l`. Without `-l`,
`/etc/profile` is not sourced and `module` is unavailable. All
`bench-a100-raven-*.sbatch` scripts already include this shebang.
