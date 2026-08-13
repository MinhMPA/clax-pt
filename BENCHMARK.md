# BENCHMARK.md — clax + clax-pt Performance & Accuracy Benchmark Plan

**Audience:** another Claude Code session / agent team picking up benchmarking work on this repo. Read top-to-bottom before doing anything; the layout is in execution order.

**Branch context:** this document lives on `benchmark/clax-pt`, a snapshot branch that combines `feat/clax-pt` (PR#9) with three independent fixes (PR#17 TE metric, PR#18 m_H mass, PR#19 BB kernel). See "Branch geometry" at the bottom for what's in each branch.

---

## 1. What we benchmark, why

Two layers, run independently because their accuracy/cost profiles differ:

### 1.1 clax — Boltzmann solver pipeline

| Stage | Module | Bottleneck |
|---|---|---|
| Background ODE | `clax/background.py` | none (~0.01 s) |
| Recombination | `clax/thermodynamics.py` | none (~0.05 s) |
| Perturbation ODE | `clax/perturbations.py` | **dominant** (~30 s on V100, ~3 s on CPU CLASS) |
| Source-line-of-sight | `clax/harmonic.py` | secondary (~2.4 s) |
| Lensing | `clax/lensing.py` | minor |
| Nonlinear corrections | `clax/nonlinear.py` (Halofit) | minor |
| Linear matter P(k) | `clax/transfer.py` | minor |

The perturbation solve is what we want to characterize most carefully — it's the gating cost for HMC.

### 1.2 clax-pt — one-loop EFTofLSS power spectra

| Stage | Module | Bottleneck |
|---|---|---|
| FFTLog decomposition | `clax/ept.py` | ~1 s (cached) |
| IR resummation (DST nw/w split) | `clax/ept.py:_ir_resummation_numpy` | NumPy at setup; not in the trace |
| P22/P13 kernel assembly | `clax/ept.py` | minor |
| Bias spectra accumulation | `clax/ept.py:_compute_bias_spectra` | minor |
| End-to-end (`compute_ept_from_clax`) | `clax/ept.py` | ~1–3 s on top of the upstream Boltzmann solve |

`compute_ept_from_clax` *requires* an upstream `MatterPerturbationResult` from `perturbations_solve_mpk`, so the full clax-pt cost is `(upstream) + (EPT)`.

---

## 2. Quick start (smoke test)

The whole benchmark suite should complete in ~10 min on CPU at the `fast` preset. Run from the repo root:

```bash
# 1. clax solver A/B (Kvaerno5 vs Rodas5; on GPU also Rodas5Batched)
python scripts/benchmark_solvers.py --n-warmup 1 --n-repeat 3

# 2. AD-vs-FD gradient cost
python scripts/benchmark_gradients.py --n-warmup 1 --n-repeat 3

# 3. clax-pt one-loop EPT (fast preset)
python scripts/benchmark_ept.py --preset fast

# 4. Lensing C_l^phiphi at three nonlinear settings + primordial BB
python scripts/benchmark_clpp.py --preset fast --l-max 2000
```

If any of these fail with `ModuleNotFoundError: clax.ept`, you're on a branch without clax-pt; only `benchmark_solvers.py` and `benchmark_gradients.py` apply there.

---

## 3. Scripts in `scripts/`

### 3.1 `benchmark_solvers.py`

| Aspect | Detail |
|---|---|
| Measures | `perturbations_solve_mpk` time with `pt_ode_solver ∈ {kvaerno5, rodas5, rosenbrock_batched}` |
| Inputs | `--n-warmup` (default 1), `--n-repeat` (default 3) |
| Output | Per-solver median time, max/mean delta_m relative error vs Kvaerno5 reference |
| Pass criteria | rodas5 vs kvaerno5 max rel err < 0.1%; speedup ≥ 1.0× CPU, ≥ 2.0× GPU (target) |
| Skip on | `rosenbrock_batched` only attempted on GPU (`platform == "gpu"`) |

### 3.2 `benchmark_gradients.py`

| Aspect | Detail |
|---|---|
| Measures | Reverse-mode `jax.grad` vs centered FD for `P(k=0.1)` w.r.t. `{h, omega_b, omega_cdm, ln10A_s, n_s}` |
| Inputs | `--n-warmup`, `--n-repeat` |
| Output | Per-parameter AD-vs-FD agreement, AD/FD time ratio, projected scaling for d=6,10,15,20 |
| Pass criteria | AD-vs-FD agreement < 1% for each parameter; AD/FD time ratio < 1.0 for d ≥ 5 (i.e. AD already wins) |

### 3.3 `benchmark_ept.py` (clax-pt only)

| Aspect | Detail |
|---|---|
| Measures | (a) Upstream BG+TH+PT compile/cached time, (b) `compute_ept_from_clax` cached forward at z=0.38, (c) multi-z scaling over `--z-list`, (d) AD gradient `d sum P_mm / d omega_b`, (e) accuracy regression vs `reference_data/classpt_z0.38_fullrange.npz` |
| Inputs | `--n-warmup`, `--n-repeat`, `--preset {fast,medium,planck}`, `--z-list 0.0,0.38,0.61,1.0` |
| Output | Per-step timings, multi-z totals, gradient time + bwd/fwd ratio, max/mean relative error for `P_mm_real` and `P_gg_real(b1=2)` at k ∈ [0.005, 0.3] h/Mpc |
| Pass criteria | EPT forward < 5 s on CPU, < 1 s on GPU (cached); P_mm/P_gg max rel err < 1% in [0.005, 0.3] h/Mpc; gradient bwd/fwd < 5× |
| Reference | `reference_data/classpt_z0.38_fullrange.npz` (Planck 2018 LCDM, z=0.38, b1=2, b4=500) |

### 3.4 `benchmark_clpp.py`

| Aspect | Detail |
|---|---|
| Measures | (a) Upstream BG+TH+full-PT compile/cached, (b) `compute_cl_pp(... nonlinear)` at "none"/"halofit"/"ept", (c) NL/linear ratio at l ∈ {100, 500, 1000, 2000}, (d) accuracy of linear C_l^pp vs `reference_data/lcdm_fiducial/cls.npz`, (e) tensor `compute_cl_bb` (post-PR#19 path) timing + ratio vs `reference_data/tensor_r01/cls_tensor.npz` |
| Inputs | `--n-warmup`, `--n-repeat`, `--preset`, `--l-max` |
| Output | Per-NL-mode timings, NL/linear ratios, BB ratios at l ∈ {2, 10, 30, 50, 80, 100, 150, 200} |
| Pass criteria | Linear C_l^pp vs CLASS < 1% at l ≤ 2500; BB ratio in [0.95, 1.05] at l ≤ 100, [0.90, 1.10] at l ∈ {150, 200} |
| Skip on | `nonlinear="ept"` raises `ValueError` on branches without clax-pt; the script catches and continues |

### 3.5 Existing scripts (kept for reference, not extended)

- `scripts/benchmark_speed.py` — per-stage pipeline timing across presets, with C_l accuracy spot-check at l ∈ {20, 100, 500, 1000}. **Use this as the headline preset comparison.**
- `scripts/benchmark_pk.py` — direct vs table-backed P(k) workflow times (cached only).
- `scripts/benchmark_pk_gradients.py` — direct-loop vs single table-backed multi-k `jax.grad`.
- `scripts/accuracy_classpt.py` — CLASS-PT accuracy test for the 9-spectrum suite (real-space + ℓ=0,2,4 multipoles for matter and galaxy). **Run this as the EPT accuracy gate; exit code 0 = all pass.**

---

## 4. Metrics & score guidelines

### 4.1 Time metrics

| Metric | Target (CPU) | Target (V100) | Target (H100) | What it tells us |
|---|---|---|---|---|
| `fit_cl` total | n/a | **≤ 50 s** | ≤ 30 s | HMC step cost; 34 s on V100 is the current baseline |
| `planck_cl` total | n/a | n/a | **≤ 600 s** | Science-grade single eval |
| Perturbation solve only | ~3–10 s | ≤ 30 s | ≤ 15 s | Dominant stage; ablate by solver choice |
| EPT forward (default) | ≤ 5 s | ≤ 2 s | ≤ 1 s | Adds linearly to upstream cost |
| C_l^pp linear | ≤ 0.5 s | ≤ 0.1 s | ≤ 0.05 s | Cheap relative to upstream |
| Gradient bwd/fwd ratio | < 5× | < 4× | < 3× | HMC gradient cost |

Times are cached (post-JIT). First-call (compile) times can be ~30–60 s on top.

### 4.2 Accuracy metrics

Pass thresholds for "the snapshot is healthy":

| Quantity | Reference | Pass threshold | Source |
|---|---|---|---|
| Linear P(k), k ∈ [0.001, 0.3] Mpc⁻¹ | `reference_data/lcdm_fiducial/pk.npz` | max < 0.5%, median < 0.1% | clax/CLASS |
| C_l^TT, ℓ ∈ [20, 2000] | `reference_data/lcdm_fiducial/cls.npz` | max < 0.5% | clax/CLASS |
| C_l^EE, ℓ ∈ [20, 2000] | same | max < 0.5% (post-PR#18 fix) | clax/CLASS |
| C_l^TE | same | use `\|TE\|/√(TT·EE) < 0.02` skip | clax/CLASS |
| C_l^pp linear, ℓ ∈ [100, 2500] | same (`pp` key) | max < 1% | clax/CLASS |
| Lensed C_l^TT/EE, ℓ ∈ [50, 2000] | `reference_data/lcdm_fiducial/cls_lensed.npz` | max < 0.5% | clax/CLASS |
| Primordial C_l^BB, ℓ ∈ [2, 200] | `reference_data/tensor_r01/cls_tensor.npz` | ratio in [0.95, 1.05] at ℓ ≤ 100 | clax/CLASS |
| EPT P_mm/P_gg/P_gm real, k ∈ [0.005, 0.3] h/Mpc | `reference_data/classpt_z0.38_fullrange.npz` | max < 1% | clax-pt/CLASS-PT |
| EPT P_mm/P_gg ℓ=0,2 | same | max < 1% | clax-pt/CLASS-PT |
| EPT P_mm/P_gg ℓ=4 | same | abs/max(ref) < 2% (zero-crossing-aware) | clax-pt/CLASS-PT |

### 4.3 Regression detection

Every benchmark script prints both timing and an accuracy spot-check. To detect regressions, save run output and diff against a baseline:

```bash
mkdir -p benchmark_results/$(date +%F)
python scripts/benchmark_solvers.py --n-warmup 2 --n-repeat 5 \
    > benchmark_results/$(date +%F)/solvers_v100.txt 2>&1
diff benchmark_results/2026-05-04/solvers_v100.txt \
     benchmark_results/$(date +%F)/solvers_v100.txt
```

Acceptable variance: ±5% on cached timings (CPU thermal, GPU SM contention), 0% on accuracy (deterministic).

---

## 5. Slurm submission templates

The repo doesn't ship cluster-specific configs — pick the cluster, fill in the placeholders.

### 5.1 GPU job (single node, 1× V100/A100/H100)

```bash
#!/bin/bash
#SBATCH --job-name=clax-bench-gpu
#SBATCH --partition=GPU                  # adjust per cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1                         # or --gres=gpu:v100:1 / gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clax-bench-gpu-%j.out
#SBATCH --error=logs/clax-bench-gpu-%j.err

# --- Environment setup (adjust per cluster) ---
module purge
module load cuda/12.4 anaconda3
source activate clax_class-pt_py310forge      # or your env name
cd $SLURM_SUBMIT_DIR

# --- Sanity ---
python -c "import jax; print(jax.devices()); print(jax.__version__)"

# --- Run benchmark suite ---
mkdir -p benchmark_results/$(date +%F)
OUT=benchmark_results/$(date +%F)/$(hostname)-gpu

python scripts/benchmark_solvers.py     --n-warmup 2 --n-repeat 5  > ${OUT}-solvers.txt    2>&1
python scripts/benchmark_gradients.py   --n-warmup 2 --n-repeat 5  > ${OUT}-gradients.txt  2>&1
python scripts/benchmark_speed.py       fit_cl                     > ${OUT}-fit_cl.txt     2>&1
python scripts/benchmark_speed.py       planck_cl                  > ${OUT}-planck_cl.txt  2>&1
python scripts/benchmark_ept.py         --preset medium            > ${OUT}-ept.txt        2>&1
python scripts/benchmark_clpp.py        --preset medium --l-max 2500 > ${OUT}-clpp.txt    2>&1
python scripts/accuracy_classpt.py                                 > ${OUT}-accuracy.txt   2>&1
```

### 5.2 CPU job (smoke regression / fallback)

```bash
#!/bin/bash
#SBATCH --job-name=clax-bench-cpu
#SBATCH --partition=RM                   # adjust per cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clax-bench-cpu-%j.out
#SBATCH --error=logs/clax-bench-cpu-%j.err

module purge
module load anaconda3
source activate clax_class-pt_py310forge
cd $SLURM_SUBMIT_DIR

# Force JAX onto CPU
export JAX_PLATFORMS=cpu

mkdir -p benchmark_results/$(date +%F)
OUT=benchmark_results/$(date +%F)/$(hostname)-cpu

python scripts/benchmark_solvers.py     --n-warmup 1 --n-repeat 3  > ${OUT}-solvers.txt    2>&1
python scripts/benchmark_gradients.py   --n-warmup 1 --n-repeat 3  > ${OUT}-gradients.txt  2>&1
python scripts/benchmark_speed.py       fit_cl                     > ${OUT}-fit_cl.txt     2>&1
python scripts/benchmark_ept.py         --preset fast              > ${OUT}-ept.txt        2>&1
python scripts/benchmark_clpp.py        --preset fast --l-max 2000 > ${OUT}-clpp.txt       2>&1
python scripts/accuracy_classpt.py                                 > ${OUT}-accuracy.txt   2>&1
```

### 5.3 Multi-cosmology accuracy scan (CPU array job)

The repo has 10+ cosmology variations under `reference_data/{h_high,h_low,omega_b_high,...}`. Compare clax against each:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=clax-multicosmo
#SBATCH --partition=RM
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/multicosmo-%A_%a.out

COSMOS=(h_high h_low omega_b_high omega_b_low omega_cdm_high omega_cdm_low \
        ns_high ns_low tau_high tau_low)
COSMO=${COSMOS[$SLURM_ARRAY_TASK_ID]}

source activate clax_class-pt_py310forge
cd $SLURM_SUBMIT_DIR
export JAX_PLATFORMS=cpu

python -m pytest tests/test_high_l.py -v -k "$COSMO" \
    > benchmark_results/$(date +%F)/multicosmo-${COSMO}.txt 2>&1
```

### 5.4 Submission

```bash
sbatch slurm/bench-gpu.sbatch        # one V100/H100 node, ~2 h
sbatch slurm/bench-cpu.sbatch        # one CPU node, ~4 h
sbatch slurm/multicosmo.sbatch       # 10× CPU jobs, ~1 h each
```

Output ends up in `benchmark_results/<DATE>/`.

---

## 6. Reporting

Each script writes plaintext to stdout. To produce a summary:

```bash
RESULTS=benchmark_results/$(date +%F)
{
  echo "# clax benchmark report — $(date)"
  echo ""
  echo "## Platform"
  for f in $RESULTS/*-solvers.txt; do
    head -3 "$f"
    echo ""
  done
  echo "## Headline numbers"
  grep -h "median\|Backward/forward\|max rel err" $RESULTS/*.txt
} > $RESULTS/REPORT.md
```

For paper-ready figures: see `notebooks/clax-pt_full_validation.ipynb` (clax-pt) and `scripts/validate_and_plot.py`.

---

## 7. Branch geometry

```
upstream/main (smsharma/clax)
└── feat/clax-pt (PR#9 draft)
    └── fix/n-H-0-mass (PR#18) ┐
    └── fix/bb-kernel-and-fine-k (PR#19) ┤
    └── docs/te-zero-crossing-metric (PR#17) ┤
                                              v
                                    benchmark/clax-pt  ← you are here
                                    (this snapshot)
```

`benchmark/clax-pt` is **not** intended to be merged upstream. It exists so a single working tree contains:

- the clax-pt module (PR#9)
- the m_H mass fix (PR#18) — closes EE ℓ=20-30 bias
- the BB kernel fix (PR#19) — closes primordial BB
- the TE metric cleanup (PR#17) — closes the misleading "TE zero-crossings" Known Limitation

When the upstream PRs eventually merge, this snapshot can be deleted; until then it's the canonical "everything works locally" state for benchmarking.

---

## 8. Known caveats (read before reporting numbers)

1. **JIT compile time dominates first-call**, especially on GPU (~30–60 s). Always discard the first call and report cached medians.
2. **GPU memory** at `planck_cl`: ~12 GB. V100-32GB is comfortable, V100-16GB requires `pt_k_chunk_size=50` or smaller.
3. **`Rodas5Batched` is a GPU-only path**. On CPU it falls back to plain `Rodas5`.
4. **`compute_ept_from_clax` requires `MatterPerturbationResult`** from `perturbations_solve_mpk` (not the full `perturbations_solve` used for C_l). Two separate upstream solves if you want both.
5. **`compute_cl_bb` (PR#19) takes a `n_k_fine` kwarg**. Default 2000 is sub-percent at ℓ ≤ 200; lower values trade accuracy for speed.
6. **`accuracy_classpt.py` is the EPT pass/fail gate**, not `benchmark_ept.py`. The latter prints accuracy as a sanity check; the former is the official 9-spectrum verification with exit-code semantics.
7. **All comparisons are against Planck 2018 best-fit ΛCDM** (`CosmoParams()` defaults). Multi-cosmology validation is via the array job in §5.3.
8. **`jax_enable_x64=True` is mandatory** — float32 is currently infeasible (numerical instability in the perturbation ODE). This is a known limitation, not something to "fix" in benchmarking.

---

## 9. What's *not* in this benchmark suite (yet)

If you have time, these would close gaps:

- **HMC step time end-to-end.** Wrap the pipeline in a NumPyro/BlackJAX log-prob and time `numpyro.infer.NUTS.sample(...)` for one step. Currently the closest proxy is `benchmark_gradients.py` × number-of-leapfrog-steps.
- **Memory footprint per stage.** None of the existing scripts measure peak memory; consider `jax.profiler.save_device_memory_profile()`.
- **TPU.** None of the scripts have been validated on TPU. Likely works but unconfirmed.
- **Hybrid k-grid for TT ℓ>1200.** Planned PR (see memory note `project_pr18_hybrid_kgrid.md`); benchmark target untracked.
- **Sparse Jacobian for the perturbation solve.** Symbolic from SymBoltz.jl design; not in clax. Would require ~weeks; benchmark target untracked.

---

## 10. Quick reference: commands by goal

| Goal | Command |
|---|---|
| Smoke test entire suite (10 min CPU) | `for s in solvers gradients ept clpp; do python scripts/benchmark_$s.py; done` |
| HMC-readiness check | `python scripts/benchmark_speed.py fit_cl && python scripts/benchmark_gradients.py` |
| Science-grade timing | `python scripts/benchmark_speed.py planck_cl` |
| EPT accuracy gate | `python scripts/accuracy_classpt.py` (exit code = pass/fail) |
| All three lensing NL paths | `python scripts/benchmark_clpp.py --preset medium --l-max 2500` |
| Solver A/B (CPU only) | `JAX_PLATFORMS=cpu python scripts/benchmark_solvers.py` |
| Solver A/B with batched (GPU) | `python scripts/benchmark_solvers.py` |
