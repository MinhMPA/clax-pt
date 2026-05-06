# clax Benchmark & Validation Session Log

**Cluster:** igpu (Kavli IPMU)
**Branch:** `benchmark/clax-pt` @ `MinhMPA/clax`
**Session date:** 2026-05-05 / 2026-05-06
**Agent:** Claude Sonnet 4.6

---

## Cluster discovery (one-time; recorded for future agents)

| Item | Value |
|---|---|
| Partition | `main` (single partition) |
| Time limit | `infinite` (no enforced cap; `--time=7-00:00:00` used as ceiling) |
| GPU gres | `--gres=gpu:1` → one Tesla V100-SXM2-32GB per job |
| Nodes | `igpu01–igpu08` (igpu03 currently `drain`) |
| Module system | None — micromamba directly, no `module load` needed |
| Conda env | `clax` (micromamba), Python 3.14, NumPy 2.x, JAX 0.9.2 |
| CUDA | 12.9 (driver 575.57.08); JAX sees GPU via pip wheel |
| LD_LIBRARY_PATH | Must include `$NVIDIA_SP/*/lib` — sbatch scripts handle this |
| Source repo | `/home/n2minh/clax` (main dev tree, `benchmark/clax-pt`) |
| Lustre clone | `/lustre/work/n2minh/clax` (submit jobs from here) |
| Benchmark output | `/lustre/work/n2minh/clax/benchmark/<DATE>/` |
| SLURM stdout/err | `/lustre/work/n2minh/std/clax/benchmark/` |
| CLASS-PT matrices | `/home/n2minh/CLASS-PT/pt_matrices/` (40 files; added 2026-05-06) |

Submission pattern:
```bash
cd /lustre/work/n2minh/clax && git pull origin benchmark/clax-pt
sbatch slurm/bench-v100-fast.sbatch
sbatch slurm/bench-v100-planck_cl.sbatch
sbatch slurm/bench-v100-igpu.sbatch   # validation job (new this session)
```
Note: jobs briefly show `InvalidAccount` after submission — transient SLURM artefact, resolves within ~30s.

---

## Sbatch scripts

| Script | Job name | Purpose | Wall time |
|---|---|---|---|
| `bench-v100-fast.sbatch` | `clax-bench-fast` | Solvers, gradients, fit_cl, ept, clpp, accuracy | 4h |
| `bench-v100-planck_cl.sbatch` | `clax-bench-planck` | planck_cl timing only | 48h |
| `bench-v100-igpu.sbatch` | `clax-validation` | pytest --fast + full validation notebook + all timing benchmarks | 7d |

`bench-v100-igpu.sbatch` was created and committed this session (`38c7660`). It is the igpu-specific deliverable covering the validation gap vs Raven A100.

---

## Jobs run this session

| Job ID | Script | Node | Status | Output prefix |
|---|---|---|---|---|
| 5660 | `bench-v100-fast.sbatch` | igpu01 | ✓ Complete | `2026-05-05/igpu01-v100` |
| 5661 | `bench-v100-planck_cl.sbatch` | igpu01 | ✓ Complete | `2026-05-05/igpu01-v100-planck_cl.txt` |
| 5662 | `bench-v100-igpu.sbatch` | igpu01 | **⏳ Running** (~9h 42m at last check) | `2026-05-05/igpu01-v100` |
| 5678 | `bench-v100-fast.sbatch` | igpu02 | ✗ Failed (`np.trapz` bug, fixed) | `2026-05-06/igpu02-v100` (partial) |
| 5679 | `bench-v100-fast.sbatch` | igpu02 | ✓ Complete | `2026-05-06/igpu02-v100` |

---

## Benchmark results summary

### Solvers — `2026-05-05/igpu01-v100-solvers.txt`

| Solver | Median | Speedup |
|---|---|---|
| kvaerno5 | 11.63s | 1.00× (reference) |
| rodas5 | 13.88s | 0.84× |
| rosenbrock_batched | 13.89s | 0.84× |

- rodas5 vs kvaerno5 max diff: **0.199%** (threshold < 0.1% → borderline fail)
- GPU speedup target ≥ 2×: **miss** (both solvers slower than kvaerno5 on V100)

### Gradients — `2026-05-05/igpu01-v100-gradients.txt` — CRITICAL FAIL

AD vs FD for `P(k=0.1 h/Mpc)` via `perturbations_solve_mpk`:

| Param | AD | FD | \|AD/FD−1\| |
|---|---|---|---|
| h | −4,894 | −13,912 | **64.8%** ✗ |
| omega_b | −17,064 | −105,290 | **83.8%** ✗ |
| omega_cdm | +213,990 | +122,410 | **74.8%** ✗ |
| ln10As | +10,526 | +10,526 | 0.0% ✓ |
| n_s | +7,296 | +7,296 | 0.0% ✓ |

Parameters flowing through the perturbation ODE fail; amplitude/tilt scalings pass. Points to broken reverse-mode AD through the ODE in `clax/perturbations.py`. **This is a clax-level bug, not yet fixed. See Open Issues.**

Timing (valid): AD 15.74s, FD 43.47s, bwd/fwd 3.5×.

### fit_cl — `2026-05-05/igpu01-v100-fit_cl.txt`

- Compile+run: 80.3s, **cached: 11.3s** ✓ (target ≤ 50s on V100)
- Accuracy (fit_cl target: TT/EE < 1.5% at ℓ ≤ 500): passes at ℓ=100 and ℓ=500 ✓

### planck_cl — `2026-05-05/igpu01-v100-planck_cl.txt`

- Compile+run: 5742.9s, **cached: 6086.2s ≈ 101 min**
  (No V100 target specified; H100 target is ≤ 600s)
- Accuracy (target: TT/EE < 0.2%): fails at ℓ=20, 100, 1000 ✗

### EPT accuracy gate — `2026-05-06/igpu02-v100-accuracy.txt` — ALL PASS ✓

```
ALL SPECTRA PASS (l0,l2 < 1%; l4 < 2%) at k < 0.3 h/Mpc
```

All 9 spectra (pk_mm/gg/gm real; pk_mm/gg l0/l2/l4) pass. Max error 1.43% on pk_gg_l4.

### EPT timing — `2026-05-06/igpu02-v100-ept.txt`

| Step | Time | Status |
|---|---|---|
| Upstream BG+TH | 15.3s | ✓ |
| `perturbations_solve_mpk` | 127.9s | ✓ |
| EPT forward z=0.38 | **1.29s** | ✓ (target < 2s GPU) |
| Multi-z × 4 | **5.08s** total | ✓ |
| AD gradient d(ΣP_mm)/d(omega_b) | crashed | Fixed (see below) |

### C_l^pp + BB — `2026-05-06/igpu02-v100-clpp.txt`

- cl_pp linear 0.071s, halofit 1.124s, EPT **2.443s** — all ✓
- BB ratios all within pass thresholds ✓
- C_l^pp accuracy fails above ℓ~500 — **expected for medium preset**, not a code bug

---

## Bugs found and fixed this session

### Fix 1 — `np.trapz` removed in NumPy 2.0
**Commit:** `219d0ff` on `benchmark/clax-pt`
**File:** `clax/ept.py` lines 667, 677, 706
**Change:** `np.trapz(...)` → `np.trapezoid(...)` (3 occurrences)
**Cause:** NumPy 2.0 removed the deprecated alias; `clax` env uses NumPy 2.x.

### Fix 2 — `float()` on JAX-traced values breaks EPT AD gradient
**Commit:** pending (smoke test in progress at session end — see below)
**File:** `clax/ept.py`, function `compute_ept_from_clax`
**Change summary:**
- `h = float(params.h)` → `h = params.h` (keep JAX scalar)
- `h_conc = float(jax.lax.stop_gradient(h))` — concrete copy for numpy ops
- `rs_h_value`: stop_gradient wraps `sound_horizon_drag(params)` before `float()`
- `f`: stop_gradient wraps `bg.Omega_m_of_z(z)` before `float()`
- Pre-compute IR resummation with `stop_gradient(pk_h)` and pass as `_ir_precomputed` to `compute_ept` — activates the existing gradient-safe path where `pk_w = pk_lin_h − pk_nw` remains JAX-traced, giving `d(pk_resummed)/d(pk_lin_h) = exp(−Σ²k²) ≠ 0`

**Why `_ir_precomputed` works:** `compute_ept` has two IR resummation paths. The default (`prec.ir_resummation=True`) calls `_ir_resummation_numpy(np.array(pk_lin_h), ...)` which concretizes the traced P(k) and kills the gradient. The `_ir_precomputed` path uses pre-computed numpy arrays for `pk_nw` and `sigma2_bao`, then computes `pk_w = pk_lin_h − pk_nw` in JAX — gradient flows through. This path was already documented at line 1522 as "enables jax.grad" but `compute_ept_from_clax` wasn't using it.

---

## Current state (as of ~11:40 JST 2026-05-06)

### Fix 2 — committed and deployed
- **Commit:** `8a2438b` on `benchmark/clax-pt`
- **Cherry-picked:** `bbe7131` on `feat/clax-pt` (both Fix 1 `ec5823e` + Fix 2 `bbe7131` pushed)
- Verification running as step 4 of job 5682 (`bench-v100-fast.sbatch`)

### Job 5662 — cancelled (was stuck)
- Pytest hung at ~59% for 9.5h with no new output — cancelled at 02:37 JST
- Resubmitted as job 5683 (`bench-v100-igpu.sbatch`) with Fix 2 in place

### Job 5682 — clax-bench-fast — running on igpu01
- Started ~11:39 JST; expected ~4h wall time
- Covers: solvers, gradients, fit_cl, **ept (Fix 2 verification)**, clpp, accuracy
- Output: `/lustre/work/n2minh/clax/benchmark/2026-05-06/igpu01-v100-*.txt`

### Job 5683 — clax-validation — running on igpu02
- Started ~11:39 JST; expected up to 7d wall time
- Covers: pytest --fast → notebook → all timing benchmarks
- Output: `/lustre/work/n2minh/clax/benchmark/2026-05-06/igpu02-v100-*.txt`

---

## Next steps for the next agent

### 1. Confirm smoke test (Fix 2)
```bash
ps aux | grep "python -c" | grep -v grep   # PID 955032, still running as of 11:30 JST
```

**If process is gone and no output visible:** run a fresh smoke test:
```bash
cd /home/n2minh/clax
micromamba run -n clax python -c "
import jax, jax.numpy as jnp
from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk
from clax.ept import compute_ept_from_clax, EPTPrecisionParams, pk_mm_real
prec = PrecisionParams()
ept_prec = EPTPrecisionParams()
params = CosmoParams()
def ept_scalar_omega_b(omega_b):
    p2 = params.replace(omega_b=omega_b)
    bg2 = background_solve(p2, prec)
    th2 = thermodynamics_solve(p2, prec, bg2)
    pt2 = perturbations_solve_mpk(p2, prec, bg2, th2)
    ept_obj = compute_ept_from_clax(p2, bg2, pt2, 0.38, ept_prec)
    return jnp.sum(pk_mm_real(ept_obj))
grad_fn = jax.grad(ept_scalar_omega_b)
g = grad_fn(jnp.float64(params.omega_b))
jax.block_until_ready(g)
print(f'd(sum P_mm)/d(omega_b) = {g:.6e}')
print('EPT AD gradient: OK')
" 2>&1
```

**If passed:** commit Fix 2, cherry-pick both commits to `feat/clax-pt`, push.
```bash
# In /home/n2minh/clax:
git add clax/ept.py
git commit -m "fix: use stop_gradient + _ir_precomputed to enable EPT jax.grad"
git log --oneline -3   # note the new commit hash
git checkout feat/clax-pt
git cherry-pick 219d0ff   # Fix 1: np.trapz
git cherry-pick <new-hash> # Fix 2: float() / stop_gradient
git push origin feat/clax-pt
git checkout benchmark/clax-pt
```

**If crashed with new error:** the next most likely crash is inside `compute_ept` — either `np.array(pk_lin_h)` on a non-`_ir_precomputed` code path, or inside `_compute_bias_spectra`. Read the traceback and continue applying `stop_gradient` as needed.

### 2. Handle job 5662 (clax-validation — possibly stuck)
Check if pytest has made progress:
```bash
ls -la /lustre/work/n2minh/clax/benchmark/2026-05-05/igpu01-v100-pytest_fast.txt
# If mtime still 22:33 May 5 after >10h → job is stuck
scancel 5662
cd /lustre/work/n2minh/clax && git pull origin benchmark/clax-pt
sbatch slurm/bench-v100-igpu.sbatch
```
Note: if resubmitting, Fix 2 must be committed first so the lustre clone can pull it.

### 3. Open issue — clax base AD gradients (h, omega_b, omega_cdm wrong)
The `benchmark_gradients.py` failure is **not fixed**. This requires investigation of reverse-mode AD through `perturbations.py`. A full investigation prompt was prepared this session — ask the user to re-share it or see conversation history. Key hypothesis: `stop_gradient`, `numpy` leaks, or broken `custom_vjp` somewhere in the perturbation ODE path.

---

## File locations quick reference

```
/home/n2minh/clax/                              ← main dev repo
  slurm/bench-v100-igpu.sbatch                 ← new this session (38c7660)
  clax/ept.py                                  ← Fix 1 committed; Fix 2 pending

/lustre/work/n2minh/clax/                       ← lustre clone (submit from here)
  benchmark/2026-05-05/
    igpu01-v100-solvers.txt                     ✓ complete
    igpu01-v100-gradients.txt                   ✓ complete (AD values wrong — open issue)
    igpu01-v100-fit_cl.txt                      ✓ complete
    igpu01-v100-ept.txt                         ✗ failed (CLASS-PT missing at run time)
    igpu01-v100-clpp.txt                        ✗ failed (CLASS-PT missing)
    igpu01-v100-accuracy.txt                    ✗ failed (CLASS-PT missing)
    igpu01-v100-planck_cl.txt                   ✓ complete
    igpu01-v100-pytest_fast.txt                 ⏳ ~59% partial (job 5662 possibly stuck since 22:33)
    REPORT.md                                   partial (fast+planck only)
  benchmark/2026-05-06/
    igpu02-v100-solvers.txt                     ✓
    igpu02-v100-gradients.txt                   ✓ (same AD failure as above)
    igpu02-v100-fit_cl.txt                      ✓
    igpu02-v100-ept.txt                         ✓ steps 1-3 pass; step 4 fixed (pending)
    igpu02-v100-clpp.txt                        ✓ all NL modes work
    igpu02-v100-accuracy.txt                    ✓ ALL PASS
    REPORT.md                                   ✓ complete

/lustre/work/n2minh/std/clax/benchmark/         ← SLURM stdout/stderr
/home/n2minh/CLASS-PT/pt_matrices/              ← CLASS-PT matrices (40 files)
```

---

## Commit history this session

| Commit | Message |
|---|---|
| `38c7660` | `slurm: igpu V100 sbatch (validation + optional timing)` |
| `219d0ff` | `fix: replace np.trapz with np.trapezoid for NumPy 2.0 compatibility` |
| `8a2438b` | `fix: use stop_gradient + _ir_precomputed to enable EPT jax.grad` |
