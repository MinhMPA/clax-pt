# ODE Step Profiling and max_steps Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile the actual number of ODE steps taken by each `PrecisionParams` preset and reduce `ode_max_steps` to the tightest safe value, cutting `RecursiveCheckpointAdjoint` checkpoint tree size and XLA compilation time proportionally.

**Architecture:** Add a one-line `jax.debug.print` inside the main diffeqsolve calls in `perturbations.py` to surface per-k-mode step counts during a dedicated SLURM profiling job. Parse the output, compute `max_observed × 3` rounded up to the next power of 2 as the new ceiling, update `params.py`, verify tests, then remove the debug instrumentation.

**Tech Stack:** JAX 0.9.2, diffrax (`sol.stats["num_steps"]`), micromamba `clax` env, SLURM on igpu V100.

---

## Background

`RecursiveCheckpointAdjoint()` (called with no `checkpoints=` argument) sizes its checkpoint tree from `max_steps`. Current values in `clax/params.py`:

| Preset | `ode_max_steps` | Notes |
|---|---|---|
| `fit_cl` | 1,024 | Code comment: "actual steps ~460" — already measured |
| `fast_cl` | 65,536 | No measured actual count |
| `medium_cl` | 65,536 | No measured actual count |
| `science_cl` | 131,072 | No measured actual count |
| `planck_cl` | 131,072 | No measured actual count |
| `planck_fast` | 65,536 | No measured actual count |

Every factor-of-2 reduction in `max_steps` removes one level from the checkpoint tree and shrinks the XLA backward graph. Going from 131,072 → 8,192 (if the ODE only needs ~3,000 steps) removes 4 tree levels and may cut the observed ~28-minute backward compilation significantly.

`sol.stats["num_steps"]` counts total steps attempted (accepted + rejected). Inside a vmapped k-batch solve it is an array of shape `(batch_size,)`. We want `max()` across all batches and all parameter points.

---

## Files

| File | Action | Purpose |
|---|---|---|
| `clax/perturbations.py` | Modify temporarily | Add `jax.debug.print` after each diffeqsolve call |
| `scripts/profile_ode_steps.py` | Create | Runs each preset × param set, captures step count output |
| `slurm/bench-v100-profile-steps.sbatch` | Create | igpu SLURM job for the profiling run |
| `clax/params.py` | Modify | Update `ode_max_steps` values after profiling |

---

## Task 1: Add step-count instrumentation to perturbations.py

**Files:**
- Modify: `clax/perturbations.py` (all lines matching `max_steps=prec.ode_max_steps`)

- [ ] **Step 1.1: Find the exact line numbers**

```bash
grep -n "max_steps=prec.ode_max_steps" clax/perturbations.py
```

Expected: 5 line numbers. Note them — each is inside a `sol = diffrax.diffeqsolve(...)` block.

- [ ] **Step 1.2: Add debug print after each diffeqsolve block**

For each block, add one line immediately after the closing `)`. Use a distinct label per call site so the output is parseable. Example for the block at ~line 2019:

```python
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_ode_rhs),
            solver=_get_stiff_solver(prec.pt_ode_solver),
            t0=tau_ini,
            t1=tau_max,
            dt0=tau_ini * 0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=tau_grid),
            stepsize_controller=_make_scalar_pid_controller(
                prec=prec, k=k, idx=idx, config=pid_config,
            ),
            adjoint=_get_adjoint(prec.ode_adjoint),
            max_steps=prec.ode_max_steps,
            args=ode_args,
        )
        jax.debug.print("[STEPS] caller=pt_scalar num_steps={s}", s=sol.stats["num_steps"])
```

Add the same pattern at all 5 call sites, using labels:
- `pt_scalar` — scalar (single k-mode) perturbation solve
- `pt_direct` — DirectAdjoint path
- `pt_mpk` — `perturbations_solve_mpk` inner loop
- `pt_tensor` — tensor perturbations
- `pt_inner` — any remaining inner solve

- [ ] **Step 1.3: Verify the edit compiles**

```bash
cd /lustre/work/n2minh/clax
micromamba run -n clax python -c \
  "from clax.perturbations import perturbations_solve_mpk; print('OK')"
```

Expected: `OK`

---

## Task 2: Write the profiling script

**Files:**
- Create: `scripts/profile_ode_steps.py`

- [ ] **Step 2.1: Create the script**

```python
#!/usr/bin/env python3
"""Profile actual ODE step counts for each PrecisionParams preset.

Run on an igpu V100 node via slurm/bench-v100-profile-steps.sbatch.
Output lines of the form:
    [STEPS] caller=pt_mpk num_steps=[4 7 5 3 ...]
are printed as side effects of jax.debug.print inside perturbations.py.

Usage (from /lustre/work/n2minh/clax):
    python scripts/profile_ode_steps.py 2>&1 | tee profile_steps_out.txt
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk

PRESETS = {
    "fit_cl":     PrecisionParams.fit_cl(),
    "fast_cl":    PrecisionParams.fast_cl(),
    "medium_cl":  PrecisionParams.medium_cl(),
    "planck_cl":  PrecisionParams.planck_cl(),
}

# 8 parameter points — fiducial + stress cases likely to take more steps
PARAM_SETS = {
    "fiducial":       CosmoParams(),
    "high_omega_b":   CosmoParams(omega_b=0.0264),   # +20%
    "low_omega_b":    CosmoParams(omega_b=0.0183),   # -20%
    "high_omega_cdm": CosmoParams(omega_cdm=0.144),  # +20%
    "low_omega_cdm":  CosmoParams(omega_cdm=0.096),  # -20%
    "high_h":         CosmoParams(h=0.741),
    "low_h":          CosmoParams(h=0.574),
    "massive_nu":     CosmoParams(m_ncdm=0.06),
}

def run_one(preset_name, prec, param_name, params):
    print(f"\n{'='*60}", flush=True)
    print(f"[PROFILE] preset={preset_name}  params={param_name}", flush=True)
    print(f"[PROFILE] ode_max_steps={prec.ode_max_steps}", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    # [STEPS] lines are emitted by jax.debug.print inside perturbations.py
    _pt = perturbations_solve_mpk(params, prec, bg, th)
    print(f"[DONE] preset={preset_name} params={param_name}", flush=True)

if __name__ == "__main__":
    print("Platform:", jax.devices(), flush=True)
    for preset_name, prec in PRESETS.items():
        for param_name, params in PARAM_SETS.items():
            try:
                run_one(preset_name, prec, param_name, params)
            except Exception as exc:
                print(f"[ERROR] {preset_name}/{param_name}: {exc}", flush=True)
    print("\n[PROFILE COMPLETE]", flush=True)
```

- [ ] **Step 2.2: Smoke-test for syntax errors (login node)**

```bash
cd /home/n2minh/clax
micromamba run -n clax python -m py_compile scripts/profile_ode_steps.py && echo "syntax OK"
```

Expected: `syntax OK`

---

## Task 3: Write sbatch, commit, and submit

**Files:**
- Create: `slurm/bench-v100-profile-steps.sbatch`

- [ ] **Step 3.1: Create the sbatch**

```bash
cat > /home/n2minh/clax/slurm/bench-v100-profile-steps.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH --job-name=clax-profile-steps
#SBATCH --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp

eval "$(micromamba shell hook --shell bash)"
micromamba activate clax

NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== Platform ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX', jax.__version__)"

cd /lustre/work/n2minh/clax
DATE=$(date +%F)
OUT=/lustre/work/n2minh/clax/benchmark/${DATE}/$(hostname)-v100-profile-steps.txt
mkdir -p "$(dirname "${OUT}")"

echo "=== ODE step profiling start: $(date) ===" | tee "${OUT}"
python scripts/profile_ode_steps.py 2>&1 | tee -a "${OUT}"
echo "=== Done: $(date) ===" | tee -a "${OUT}"
SBATCH
```

- [ ] **Step 3.2: Commit everything**

```bash
cd /home/n2minh/clax
git add clax/perturbations.py scripts/profile_ode_steps.py \
        slurm/bench-v100-profile-steps.sbatch \
        docs/superpowers/plans/2026-05-06-profile-max-steps.md
git commit -m "chore: add ODE step profiling instrumentation and sbatch"
git push origin benchmark/clax-pt
```

- [ ] **Step 3.3: Pull lustre clone and submit**

```bash
cd /lustre/work/n2minh/clax && git pull origin benchmark/clax-pt
sbatch slurm/bench-v100-profile-steps.sbatch
```

Expected: job appears in `squeue -u n2minh` within 30 s (ignore transient `InvalidAccount`).
Wall time estimate: ~2–5 h (4 presets × 8 param sets × one compile per preset).

---

## Task 4: Parse results and compute new max_steps values

*Do this after the SLURM job from Task 3 completes.*

**Files:**
- Modify: `clax/params.py` (one `ode_max_steps=` line per preset)

- [ ] **Step 4.1: Find the output file and extract per-preset maxima**

```bash
OUT=$(ls -t /lustre/work/n2minh/clax/benchmark/*/igpu*-v100-profile-steps.txt | head -1)

# Print max num_steps per preset block
python3 - << 'PY'
import re, sys

out_file = open("/lustre/work/n2minh/clax/benchmark/FILL_DATE/FILL_NODE-v100-profile-steps.txt").read()

current_preset = None
maxima = {}
for line in out_file.splitlines():
    m = re.search(r"\[PROFILE\] preset=(\S+)", line)
    if m:
        current_preset = m.group(1)
    m = re.search(r"\[STEPS\].*num_steps=\[([^\]]+)\]", line)
    if m and current_preset:
        vals = [int(x) for x in m.group(1).split()]
        maxima.setdefault(current_preset, 0)
        maxima[current_preset] = max(maxima[current_preset], max(vals))

import math
def next_pow2(x):
    return 2 ** math.ceil(math.log2(max(x, 1)))

print(f"{'Preset':<12} {'max_observed':>14} {'×3':>8} {'next_pow2':>10} {'log2':>6}")
print("-" * 55)
for preset, mx in sorted(maxima.items()):
    target = next_pow2(mx * 3)
    print(f"{preset:<12} {mx:>14} {mx*3:>8} {target:>10} {math.log2(target):>6.0f}")
PY
```

(Replace `FILL_DATE` and `FILL_NODE` with the actual path.)

- [ ] **Step 4.2: Update params.py with the computed ceilings**

Open `clax/params.py` and update each `ode_max_steps=` line. Add a comment with the profiling evidence. Example (fill in actual numbers):

```python
# fit_cl
ode_max_steps=2048,   # profiled max ~460 steps; 460×3=1380 → 2048 (was 1024)

# fast_cl
ode_max_steps=????,   # profiled max ~???? steps; ????×3=???? → ???? (was 65536)

# medium_cl
ode_max_steps=????,   # profiled max ~???? steps; ????×3=???? → ???? (was 65536)

# planck_cl / science_cl
ode_max_steps=????,   # profiled max ~???? steps; ????×3=???? → ???? (was 131072)
```

Rule: if the new value is *larger* than the current value for any preset (meaning the ODE actually needs more steps than the ceiling allows), keep the current value and investigate why.

---

## Task 5: Verify correctness

- [ ] **Step 5.1: Commit params.py change and pull on lustre**

```bash
cd /home/n2minh/clax
git add clax/params.py
git commit -m "perf: reduce ode_max_steps from profiling data (placeholder values)"
git push origin benchmark/clax-pt
cd /lustre/work/n2minh/clax && git pull origin benchmark/clax-pt
```

- [ ] **Step 5.2: Submit pytest verification job**

```bash
sbatch --wrap="
eval \"\$(micromamba shell hook --shell bash)\"
micromamba activate clax
NVIDIA_SP=\$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia
for d in \"\${NVIDIA_SP}\"/*/lib; do export LD_LIBRARY_PATH=\"\${d}:\${LD_LIBRARY_PATH:-}\"; done
cd /lustre/work/n2minh/clax
python -m pytest tests/ --fast -q 2>&1 | tee benchmark/\$(date +%F)/pytest-max-steps-check.txt
" \
  --job-name=clax-pytest-check \
  --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j \
  --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j \
  --gres=gpu:1 --mem=125GB --time=04:00:00 \
  --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
```

- [ ] **Step 5.3: Check pytest result**

```bash
tail -5 /lustre/work/n2minh/clax/benchmark/$(date +%F)/pytest-max-steps-check.txt
```

Expected: identical or better pass count vs baseline (231 passed, 2 pre-existing failures for massive-nu + Rosenbrock). If new "maximum number of solver steps was reached" failures appear, double that preset's `ode_max_steps` and repeat from Step 5.1.

---

## Task 6: Measure compilation time improvement and clean up

- [ ] **Step 6.1: Submit bench-v100-fast.sbatch to measure new compile time**

```bash
cd /lustre/work/n2minh/clax
sbatch slurm/bench-v100-fast.sbatch
```

- [ ] **Step 6.2: Compare EPT backward compile time to baseline**

Pre-change baseline (2026-05-06 igpu01): gradient compile = **1679 s**, median eval = 420.9 s.

```bash
grep "Gradient compile\|median" \
  /lustre/work/n2minh/clax/benchmark/$(date +%F)/*-ept.txt
```

Record the new compile time. The reduction should be roughly proportional to the reduction in `log2(max_steps)` for the preset used by `benchmark_ept.py` (`medium_cl`).

- [ ] **Step 6.3: Remove the jax.debug.print lines from perturbations.py**

```bash
grep -n "STEPS" /home/n2minh/clax/clax/perturbations.py
```

Delete every matching line. Verify:

```bash
grep "STEPS" /home/n2minh/clax/clax/perturbations.py
# Expected: no output
```

- [ ] **Step 6.4: Remove the profiling sbatch (keep the script for future use)**

The profiling script is reusable. The sbatch can stay. Update the commit message in the final commit to include actual measured values.

- [ ] **Step 6.5: Final commit**

```bash
cd /home/n2minh/clax
git add clax/perturbations.py clax/params.py
git commit -m "$(cat <<'EOF'
perf: reduce ode_max_steps from profiling; remove debug instrumentation

Profiled actual ODE step counts across 8 parameter sets × 4 presets
on V100 (igpu, 2026-05-06). Results (max_observed × 3 → next_pow2):

  fit_cl:    ~460 steps → 2048   (was 1,024)
  fast_cl:   ~???? steps → ????  (was 65,536)
  medium_cl: ~???? steps → ????  (was 65,536)
  planck_cl: ~???? steps → ????  (was 131,072)

Smaller max_steps shrinks the RecursiveCheckpointAdjoint checkpoint tree
(O(log max_steps) depth), reducing XLA backward graph size and jax.grad
compilation time. Measured improvement on medium_cl: ???? s → ???? s.
EOF
)"
git push origin benchmark/clax-pt
```

---

## Also recommended (separate task, not in scope here)

Add the JAX persistent compilation cache to every benchmark script and `conftest.py`. Pay the XLA compile cost once, reuse across all subsequent jobs:

```python
# Top of every scripts/benchmark_*.py and tests/conftest.py
import jax
jax.experimental.compilation_cache.initialize_cache(
    "/lustre/work/n2minh/jax-cache"
)
```

Together with reduced `max_steps`, this eliminates most of the ~190 minutes of hidden compilation time currently paid on every `bench-v100-igpu.sbatch` run.
