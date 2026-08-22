# Benchmark Analysis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage pipeline that automatically detects arriving benchmark result files, parses every key metric, checks against BENCHMARK.md §4 thresholds, and produces a full REPORT.md + ANALYSIS.md with a Claude API narrative — firing a push notification before the user reaches their laptop.

**Architecture:** Stage 1 (`analyze_benchmark.py`) is a pure-stdlib deterministic parser that writes `results.json` + `REPORT.md`. Stage 2 (`agent_analyze_benchmark.py`) uses the Anthropic SDK to reason over the parsed JSON and raw files, producing `ANALYSIS.md` and a push notification. A watcher script (`watch_benchmark.py`) runs on a 30-min cron, detects new files, drives both stages, and self-cancels when all 7 result files have been analyzed.

**Tech Stack:** Python 3.11 stdlib (`re`, `json`, `pathlib`, `argparse`, `subprocess`), `anthropic` 0.98.1 (already installed in `py311forge`), Claude Code `PushNotification` + `CronCreate`/`CronDelete` tools.

---

## File map

| File | Role |
|---|---|
| `scripts/analyze_benchmark.py` | Stage 1: parse 7 .txt files → `results.json` + `REPORT.md` |
| `scripts/agent_analyze_benchmark.py` | Stage 2: Claude API agent → `ANALYSIS.md` + push notification |
| `scripts/watch_benchmark.py` | Watcher: polls result dir, drives Stage 1+2, self-cancels |
| `tests/test_analyze_benchmark.py` | Unit tests for all Stage 1 parsers and threshold checker |
| `tests/fixtures/benchmark/solvers.txt` | Fixture: realistic solver output |
| `tests/fixtures/benchmark/gradients.txt` | Fixture: realistic gradients output |
| `tests/fixtures/benchmark/fit_cl.txt` | Fixture: realistic fit_cl output |
| `tests/fixtures/benchmark/ept.txt` | Fixture: realistic EPT output |
| `tests/fixtures/benchmark/clpp.txt` | Fixture: realistic clpp output |
| `tests/fixtures/benchmark/accuracy.txt` | Fixture: realistic accuracy output (all PASS) |
| `tests/fixtures/benchmark/accuracy_fail.txt` | Fixture: accuracy output with one FAIL |
| `tests/fixtures/benchmark/planck_cl.txt` | Fixture: realistic planck_cl output |

---

## Task 1: Test fixtures

**Files:**
- Create: `tests/fixtures/benchmark/` (8 fixture files)
- Create: `tests/test_analyze_benchmark.py` (skeleton only, imports)

- [ ] **Step 1: Create the fixtures directory**

```bash
mkdir -p tests/fixtures/benchmark
```

- [ ] **Step 2: Write `tests/fixtures/benchmark/solvers.txt`**

```
Platform: gpu ([CudaDevice(id=0)])
JAX version: 0.10.0

Computing background + thermodynamics (shared)...
  conformal age = 14153.25

--- kvaerno5 ---
  n_k=89, rtol=0.0001, atol=0.0001
  Time (median of 5): 13.09s
  delta_m shape: (89,)
  (reference for accuracy comparison)

--- rodas5 ---
  n_k=89, rtol=0.0001, atol=0.0001
  Time (median of 5): 6.21s
  delta_m shape: (89,)
  vs kvaerno5: max=0.0872%, mean=0.0312%

--- rosenbrock_batched ---
  n_k=89, rtol=0.0001, atol=0.0001
  Time (median of 5): 6.19s
  delta_m shape: (89,)
  vs kvaerno5: max=0.0872%, mean=0.0312%

============================================================
SUMMARY
============================================================
  Solver                    Median (s)    Speedup
  kvaerno5                       13.09      1.00x
  rodas5                          6.21      2.11x
  rosenbrock_batched              6.19      2.11x
```

- [ ] **Step 3: Write `tests/fixtures/benchmark/gradients.txt`**

```
Platform: gpu ([CudaDevice(id=0)])
JAX version: 0.10.0

Objective: P(k=0.1 h/Mpc)
Parameters: ['h', 'omega_b', 'omega_cdm', 'ln10A_s', 'n_s'] (d=5)
Solver: kvaerno5

Forward evaluation:
  P(k) = 2.0183e+04
  First call: 45.23s (compile + run)
  Cached:     13.09s

AD gradient (reverse-mode, jax.grad):
  Median time: 28.34s (5 repeats)
  Backward/forward ratio: 2.2x
    dP/d(h) = -1.2345e+05
    dP/d(omega_b) = 3.4567e+04
    dP/d(omega_cdm) = -8.9012e+03
    dP/d(ln10A_s) = 2.0183e+04
    dP/d(n_s) = 1.5678e+04

FD gradient (centered, 2*5=10 evaluations):
  Median time: 134.50s (5 repeats)
  Per-evaluation: 13.45s
    dP/d(h) = -1.2350e+05
    dP/d(omega_b) = 3.4570e+04
    dP/d(omega_cdm) = -8.9015e+03
    dP/d(ln10A_s) = 2.0185e+04
    dP/d(n_s) = 1.5680e+04

============================================================
COMPARISON
============================================================
  Forward:  13.09s
  AD grad:  28.34s  (1 backward pass)
  FD grad:  134.50s  (10 forward passes)
  Speedup:  4.7x  (AD vs FD)
  Effective backward cost: 1.2x forward

AD vs FD agreement:
  Param           AD            FD       |AD/FD-1|
  h        -1.2345e+05  -1.2350e+05     0.0405%
  omega_b   3.4567e+04   3.4570e+04     0.0087%
  omega_cdm -8.9012e+03 -8.9015e+03     0.0034%
  ln10A_s   2.0183e+04   2.0185e+04     0.0099%
  n_s       1.5678e+04   1.5680e+04     0.0128%

Projected scaling:
  d= 5: FD= 134.5s, AD=  28.3s, ratio 0.21x (AD wins)
  d= 6: FD= 161.4s, AD=  28.3s, ratio 0.18x (AD wins)
  d=10: FD= 269.0s, AD=  28.3s, ratio 0.11x (AD wins)
  d=15: FD= 403.5s, AD=  28.3s, ratio 0.07x (AD wins)
  d=20: FD= 538.0s, AD=  28.3s, ratio 0.05x (AD wins)
```

- [ ] **Step 4: Write `tests/fixtures/benchmark/fit_cl.txt`**

```
GPU: [CudaDevice(id=0)]
JAX version: 0.10.0

============================================================
BENCHMARK: fit_cl
============================================================

First call:
  Background:        0.1s
  Thermodynamics:    0.3s
  Perturbations:    32.1s
  Harmonic:          2.4s
  TOTAL (1st):      34.9s

Second call (cached):
  Background:        0.1s
  Thermodynamics:    0.3s
  Perturbations:    30.2s
  Harmonic:          2.4s
  TOTAL (2nd):      33.0s

Accuracy vs CLASS (fiducial, massless ncdm, RECFAST):
    l    TT err%    EE err%    TE err%
   20    -0.123    +0.456    +0.789
  100    -0.012    +0.034    -0.056
  500    -0.001    +0.002    -0.003
 1000    -0.001    +0.001    +1.700
```

- [ ] **Step 5: Write `tests/fixtures/benchmark/ept.txt`**

```
Platform: gpu ([CudaDevice(id=0)])
JAX version: 0.10.0
Preset: medium, z list: [0.0, 0.38, 0.61, 1.0]

Step 1 — upstream Boltzmann (BG + TH + perturbations_solve_mpk)
  BG + TH (compile + cached): 1.23s
  perturbations_solve_mpk (compile + cached): 32.10s
  pt.delta_m shape: (89, 500)

Step 2 — EPT forward at z=0.38 (default EPTPrecisionParams, nmax=256)
  median: 0.412s   range: [0.401, 0.425]
  k-grid: 256 points, kh=[0.0001, 100.0] h/Mpc
  P_mm(k=0.1): 10234.56 (Mpc/h)^3

Step 3 — multi-z scaling, z_list = [0.0, 0.38, 0.61, 1.0]
  z= 0.00: median 0.405s
  z= 0.38: median 0.412s
  z= 0.61: median 0.408s
  z= 1.00: median 0.410s
  Total over 4 z values: 1.635s (avg 0.409s per z)

Step 4 — AD gradient: d(sum P_gg) / d(omega_b)
  Gradient compile (first call): 89.34s
  median: 1.234s   range: [1.221, 1.249]
  Backward/Forward ratio (full pipeline): 0.37x

Step 5 — accuracy regression vs reference_data/classpt_z0.38_fullrange.npz
  P_mm real: max rel err = 0.312%, mean = 0.041%
  P_gg real (b1=2): max rel err = 0.312%, mean = 0.041%

============================================================
SUMMARY
============================================================
  Preset:                   medium
  Platform:                 gpu
  Upstream BG+TH+PT (compile + cached): 33.33s
  EPT forward at z=0.38 (cached):  0.412s
  EPT forward (multi-z avg):       0.409s
  EPT scalar gradient (full pipe): 1.234s
```

- [ ] **Step 6: Write `tests/fixtures/benchmark/clpp.txt`**

```
Platform: gpu ([CudaDevice(id=0)])
JAX version: 0.10.0
Preset: medium, l_max: 2500

Step 1 — upstream Boltzmann (BG + TH + perturbations_solve)
  BG + TH: 0.41s
  perturbations_solve: 31.2s
  source_phi_plus_psi shape: (89, 500)

Step 2.none — compute_cl_pp(nonlinear='none', l_max=2500)
  median: 0.087s   range: [0.082, 0.093]
  C_l^pp(l=100) = 4.5678e-07
  C_l^pp(l=1000) = 1.2345e-08

Step 2.halofit — compute_cl_pp(nonlinear='halofit', l_max=2500)
  median: 0.234s   range: [0.229, 0.241]
  C_l^pp(l=100) = 4.6789e-07
  C_l^pp(l=1000) = 1.3456e-08

Step 2.ept — compute_cl_pp(nonlinear='ept', l_max=2500)
  median: 0.891s   range: [0.880, 0.903]
  C_l^pp(l=100) = 4.6901e-07
  C_l^pp(l=1000) = 1.3567e-08

Ratio  C_l^pp[halofit] / C_l^pp[none]:
  l= 100: 1.0243
  l= 500: 1.0512
  l=1000: 1.0902
  l=2000: 1.1234

Ratio  C_l^pp[ept] / C_l^pp[none]:
  l= 100: 1.0267
  l= 500: 1.0534
  l=1000: 1.0934
  l=2000: 1.1267

Step 4 — C_l^pp accuracy vs CLASS reference (nonlinear='none'):
      l |        clax |       CLASS |  rel diff
  --------------------------------------------------
    100 |  4.5678e-07 |  4.5701e-07 |   -0.050%
    500 |  2.3456e-07 |  2.3478e-07 |   -0.094%
   1000 |  1.2345e-08 |  1.2356e-08 |   -0.089%
   1500 |  3.4567e-09 |  3.4589e-09 |   -0.064%
   2000 |  1.5678e-09 |  1.5690e-09 |   -0.076%
   2500 |  8.9012e-10 |  8.9123e-10 |   -0.125%

Step 5 — primordial C_l^BB (compute_cl_bb, post-PR#19 path)
  tensor_perturbations_solve: 4.56s
  compute_cl_bb (cached, n_k_fine=2000): 0.234s
     l |        clax |       CLASS |   ratio
  ---------------------------------------------
     2 |  1.2345e-15 |  1.2367e-15 |  0.9982
    10 |  3.4567e-15 |  3.4612e-15 |  0.9987
    30 |  7.8901e-15 |  7.9023e-15 |  0.9985
    50 |  9.0123e-15 |  9.0189e-15 |  0.9993
    80 |  8.9012e-15 |  8.9056e-15 |  0.9995
   100 |  7.8901e-15 |  7.8934e-15 |  0.9996
   150 |  5.6789e-15 |  5.6823e-15 |  0.9994
   200 |  3.4567e-15 |  3.4601e-15 |  0.9990

============================================================
SUMMARY
============================================================
  Preset: medium
  Platform: gpu
  Upstream solve (BG+TH+PT): 31.61s
  compute_cl_pp('none', l_max=2500): 0.087s
  compute_cl_pp('halofit', l_max=2500): 0.234s
  compute_cl_pp('ept', l_max=2500): 0.891s
  Tensor solve + compute_cl_bb (n_k_fine=2000): 4.794s
```

- [ ] **Step 7: Write `tests/fixtures/benchmark/accuracy.txt` (all PASS)**

```
Reference: z=0.38, h=0.6736, f=0.4690
k grid: 256 points [0.0001, 100.0] h/Mpc
Bias: b1=2, b2=0, bG2=0, bGamma3=0
EFT:  cs0=0, cs2=0, cs4=0, Pshot=0

Running compute_ept ...
  Sigma_BAO^2 check (expect ~30-50 (Mpc/h)^2): 42.3 (Mpc/h)^2
  Pk_tree range: [12.3, 45678.9]
  Pk_loop range: [-234.5, 1234.5]

Accuracy at k < 0.3 h/Mpc (89 modes):

Spectrum          max_err  mean_err    k@max      pass        metric
----------------------------------------------------------------------
  pk_mm_real         0.31%     0.04%   0.0534  ✓ PASS           rel
  pk_gg_real         0.31%     0.04%   0.0534  ✓ PASS           rel
  pk_gm_real         0.31%     0.04%   0.0534  ✓ PASS           rel
  pk_mm_l0           0.59%     0.40%   0.2345  ✓ PASS           rel
  pk_mm_l2           0.70%     0.44%   0.2456  ✓ PASS           rel
  pk_mm_l4           0.70%     0.15%   0.2567  ✓ PASS   abs/max(ref)
  pk_gg_l0           0.56%     0.39%   0.2234  ✓ PASS           rel
  pk_gg_l2           0.89%     0.50%   0.2345  ✓ PASS           rel
  pk_gg_l4           1.43%     0.37%   0.2456  ✓ PASS   abs/max(ref)

--- Diagnostic: 5-point comparison at k = 0.05, 0.1, 0.15, 0.2, 0.25 h/Mpc ---

  pk_mm_real:
         k            clax        CLASS-PT     rel_err
    0.0501      10234.567      10230.456       0.040%
    0.1003       5678.901       5677.123       0.031%
    0.1501       2345.678       2344.901       0.033%
    0.2002        987.654        986.543       0.113%
    0.2503        456.789        456.234       0.122%

==================================================
ALL SPECTRA PASS (l0,l2 < 1%; l4 < 2%) at k < 0.3 h/Mpc  ✓
==================================================
```

- [ ] **Step 8: Write `tests/fixtures/benchmark/accuracy_fail.txt` (one FAIL)**

Same as above but replace the `pk_gg_l4` line with:
```
  pk_gg_l4           2.15%     0.89%   0.2456  ✗ FAIL   abs/max(ref)
```
and replace the final block with:
```
==================================================
FAILING spectra (1): pk_gg_l4
Check diagnostic output above for clues.
==================================================
```

- [ ] **Step 9: Write `tests/fixtures/benchmark/planck_cl.txt`**

Same structure as `fit_cl.txt` but with `planck_cl` preset and slower times:
```
GPU: [CudaDevice(id=0)]
JAX version: 0.10.0

============================================================
BENCHMARK: planck_cl
============================================================

First call:
  Background:        0.1s
  Thermodynamics:    0.4s
  Perturbations:   487.3s
  Harmonic:         12.1s
  TOTAL (1st):     499.9s

Second call (cached):
  Background:        0.1s
  Thermodynamics:    0.3s
  Perturbations:   485.2s
  Harmonic:         11.8s
  TOTAL (2nd):     497.4s

Accuracy vs CLASS (fiducial, massless ncdm, RECFAST):
    l    TT err%    EE err%    TE err%
   20    -0.023    +0.045    +0.089
  100    -0.002    +0.003    -0.005
  500    -0.001    +0.001    -0.001
 1000    -0.001    +0.001    +0.200
```

- [ ] **Step 10: Create test skeleton**

```python
# tests/test_analyze_benchmark.py
import json
import sys
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 11: Commit fixtures**

```bash
git add tests/fixtures/benchmark/ tests/test_analyze_benchmark.py
git commit -m "test: add benchmark analysis fixtures and test skeleton"
```

---

## Task 2: Stage 1 — core data types and file discovery

**Files:**
- Create: `scripts/analyze_benchmark.py`
- Modify: `tests/test_analyze_benchmark.py`

- [ ] **Step 1: Write failing tests for `find_result_files`**

Add to `tests/test_analyze_benchmark.py`:
```python
from scripts.analyze_benchmark import find_result_files, MetricStatus

def test_find_result_files_all_present(tmp_path):
    for name in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text("dummy")
    found, missing = find_result_files(tmp_path)
    assert set(found.keys()) == {"solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"}
    assert missing == []

def test_find_result_files_partial(tmp_path):
    for name in ["solvers", "gradients", "fit_cl"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text("dummy")
    found, missing = find_result_files(tmp_path)
    assert set(found.keys()) == {"solvers", "gradients", "fit_cl"}
    assert set(missing) == {"ept", "clpp", "accuracy", "planck_cl"}

def test_metric_status_values():
    assert MetricStatus.PASS == "PASS"
    assert MetricStatus.FAIL == "FAIL"
    assert MetricStatus.WARN == "WARN"
    assert MetricStatus.MISSING == "MISSING"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /u/minh/clax
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py::test_find_result_files_all_present tests/test_analyze_benchmark.py::test_find_result_files_partial tests/test_analyze_benchmark.py::test_metric_status_values -v
```
Expected: `ModuleNotFoundError` or `ImportError` — file doesn't exist yet.

- [ ] **Step 3: Implement core data types and `find_result_files`**

Create `scripts/analyze_benchmark.py`:
```python
"""Stage 1: parse benchmark .txt outputs and check BENCHMARK.md §4 thresholds.

Usage:
    python scripts/analyze_benchmark.py [--date YYYY-MM-DD] [--dir /path/to/results]

Writes:
    <dir>/results.json   structured metrics + pass/fail
    <dir>/REPORT.md      human-readable tables

Exit codes:
    0  all present metrics PASS
    1  one or more metrics FAIL
    2  one or more expected files MISSING
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

EXPECTED_SCRIPTS = ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]


class MetricStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    MISSING = "MISSING"


def find_result_files(result_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Return (found: {script_name: path}, missing: [script_name])."""
    found = {}
    for path in result_dir.glob("*-a100-*.txt"):
        for name in EXPECTED_SCRIPTS:
            if path.name.endswith(f"-{name}.txt") and name not in found:
                found[name] = path
    missing = [s for s in EXPECTED_SCRIPTS if s not in found]
    return found, missing
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py::test_find_result_files_all_present tests/test_analyze_benchmark.py::test_find_result_files_partial tests/test_analyze_benchmark.py::test_metric_status_values -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_benchmark.py tests/test_analyze_benchmark.py
git commit -m "feat: Stage 1 skeleton — MetricStatus and find_result_files"
```

---

## Task 3: Stage 1 — per-script parsers

**Files:**
- Modify: `scripts/analyze_benchmark.py`
- Modify: `tests/test_analyze_benchmark.py`

- [ ] **Step 1: Write failing tests for all parsers**

Add to `tests/test_analyze_benchmark.py`:
```python
from scripts.analyze_benchmark import (
    parse_solvers, parse_gradients, parse_speed,
    parse_ept, parse_clpp, parse_accuracy,
)

def _fixture(name):
    return (FIXTURES / name).read_text()

def test_parse_solvers():
    m = parse_solvers(_fixture("solvers.txt"))
    assert abs(m["kvaerno5_median_s"] - 13.09) < 0.01
    assert abs(m["rodas5_median_s"] - 6.21) < 0.01
    assert abs(m["rodas5_speedup"] - 2.11) < 0.01
    assert abs(m["rosenbrock_median_s"] - 6.19) < 0.01
    assert abs(m["rodas5_vs_kv_max_pct"] - 0.0872) < 0.001
    assert m["platform"] == "gpu"

def test_parse_gradients():
    m = parse_gradients(_fixture("gradients.txt"))
    assert abs(m["fwd_cached_s"] - 13.09) < 0.01
    assert abs(m["ad_median_s"] - 28.34) < 0.01
    assert abs(m["bwd_fwd_ratio"] - 2.2) < 0.01
    assert abs(m["ad_fd_speedup"] - 4.7) < 0.01
    assert abs(m["ad_fd_agreement"]["h"] - 0.0405) < 0.001
    assert abs(m["ad_fd_agreement"]["omega_b"] - 0.0087) < 0.001

def test_parse_speed_fit_cl():
    m = parse_speed(_fixture("fit_cl.txt"))
    assert m["preset"] == "fit_cl"
    assert abs(m["total_cached_s"] - 33.0) < 0.1
    assert abs(m["pt_cached_s"] - 30.2) < 0.1
    assert abs(m["cl_errs"][20]["tt"] - (-0.123)) < 0.001
    assert abs(m["cl_errs"][1000]["ee"] - 0.001) < 0.001

def test_parse_ept():
    m = parse_ept(_fixture("ept.txt"))
    assert abs(m["ept_fwd_cached_s"] - 0.412) < 0.001
    assert abs(m["bwd_fwd_ratio"] - 0.37) < 0.01
    assert abs(m["pmm_max_pct"] - 0.312) < 0.001
    assert abs(m["pgg_max_pct"] - 0.312) < 0.001
    assert abs(m["grad_cached_s"] - 1.234) < 0.001

def test_parse_clpp():
    m = parse_clpp(_fixture("clpp.txt"))
    assert abs(m["clpp_none_s"] - 0.087) < 0.001
    assert abs(m["clpp_halofit_s"] - 0.234) < 0.001
    assert abs(m["clpp_ept_s"] - 0.891) < 0.001
    # Step 4: max of abs rel diffs across all probe ells
    assert m["clpp_linear_max_abspct"] < 0.2
    # BB ratios: all near 1.0
    assert abs(m["bb_ratios"][2] - 0.9982) < 0.001
    assert abs(m["bb_ratios"][100] - 0.9996) < 0.001
    assert abs(m["bb_ratios"][200] - 0.9990) < 0.001

def test_parse_accuracy_pass():
    m = parse_accuracy(_fixture("accuracy.txt"))
    assert m["all_pass"] is True
    assert abs(m["spectra"]["pk_mm_real"]["max_pct"] - 0.31) < 0.01
    assert m["spectra"]["pk_mm_real"]["pass"] is True
    assert len(m["spectra"]) == 9

def test_parse_accuracy_fail():
    m = parse_accuracy(_fixture("accuracy_fail.txt"))
    assert m["all_pass"] is False
    assert m["spectra"]["pk_gg_l4"]["pass"] is False
    assert abs(m["spectra"]["pk_gg_l4"]["max_pct"] - 2.15) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -k "parse" -v
```
Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Implement `parse_solvers`**

Add to `scripts/analyze_benchmark.py`:
```python
def parse_solvers(text: str) -> dict:
    """Parse benchmark_solvers.py output."""
    m = {}
    # Platform
    plat = re.search(r"Platform: (\w+)", text)
    m["platform"] = plat.group(1) if plat else None

    # Per-solver rel err lines (before SUMMARY)
    rel = re.findall(r"vs kvaerno5: max=(\d+\.\d+)%, mean=(\d+\.\d+)%", text)
    # Summary table: name  median  speedup
    rows = re.findall(r"^\s+(kvaerno5|rodas5|rosenbrock_batched)\s+(\d+\.\d+)\s+(\d+\.\d+)x",
                      text, re.MULTILINE)
    for row in rows:
        name, median, speedup = row
        m[f"{name}_median_s"] = float(median)
        m[f"{name}_speedup"] = float(speedup)

    if len(rel) >= 1:
        m["rodas5_vs_kv_max_pct"] = float(rel[0][0])
    if len(rel) >= 2:
        m["rosenbrock_vs_kv_max_pct"] = float(rel[1][0])
    return m
```

- [ ] **Step 4: Implement `parse_gradients`**

```python
def parse_gradients(text: str) -> dict:
    """Parse benchmark_gradients.py output."""
    m = {}
    fwd = re.search(r"Cached:\s+(\d+\.\d+)s", text)
    m["fwd_cached_s"] = float(fwd.group(1)) if fwd else None

    ad = re.search(r"Median time:\s+(\d+\.\d+)s\s+\(\d+ repeats\)", text)
    m["ad_median_s"] = float(ad.group(1)) if ad else None

    bwdfwd = re.search(r"Backward/forward ratio:\s+(\d+\.\d+)x", text)
    m["bwd_fwd_ratio"] = float(bwdfwd.group(1)) if bwdfwd else None

    speedup = re.search(r"Speedup:\s+(\d+\.\d+)x\s+\(AD vs FD\)", text)
    m["ad_fd_speedup"] = float(speedup.group(1)) if speedup else None

    fd = re.search(r"FD grad:\s+(\d+\.\d+)s", text)
    m["fd_median_s"] = float(fd.group(1)) if fd else None

    # Per-parameter agreement: "  h        ...   0.0405%"
    params = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
    m["ad_fd_agreement"] = {}
    for p in params:
        pat = rf"^\s+{re.escape(p)}\s+[\d.e+-]+\s+[\d.e+-]+\s+(\d+\.\d+)%"
        hit = re.search(pat, text, re.MULTILINE)
        if hit:
            m["ad_fd_agreement"][p] = float(hit.group(1))
    return m
```

- [ ] **Step 5: Implement `parse_speed`**

```python
def parse_speed(text: str) -> dict:
    """Parse benchmark_speed.py output (fit_cl or planck_cl)."""
    m = {}
    preset = re.search(r"BENCHMARK:\s+(\w+)", text)
    m["preset"] = preset.group(1) if preset else None

    # Find "Second call (cached):" block
    cached_block = re.search(
        r"Second call \(cached\):(.*?)(?=\n\n|\Z)", text, re.DOTALL)
    if cached_block:
        block = cached_block.group(1)
        for stage in ["Background", "Thermodynamics", "Perturbations", "Harmonic"]:
            hit = re.search(rf"{stage}:\s+(\d+\.\d+)s", block)
            key = stage.lower() + "_cached_s"
            m[key] = float(hit.group(1)) if hit else None
        total = re.search(r"TOTAL \(2nd\):\s+(\d+\.\d+)s", block)
        m["total_cached_s"] = float(total.group(1)) if total else None
        m["pt_cached_s"] = m.get("perturbations_cached_s")

    # Accuracy table: "   20  -0.123   +0.456   +0.789"
    m["cl_errs"] = {}
    for hit in re.finditer(
        r"^\s+(\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)",
        text, re.MULTILINE
    ):
        l = int(hit.group(1))
        m["cl_errs"][l] = {
            "tt": float(hit.group(2)),
            "ee": float(hit.group(3)),
            "te": float(hit.group(4)),
        }
    return m
```

- [ ] **Step 6: Implement `parse_ept`**

```python
def parse_ept(text: str) -> dict:
    """Parse benchmark_ept.py output."""
    m = {}
    fwd = re.search(r"EPT forward at z=0\.38 \(cached\):\s+(\d+\.\d+)s", text)
    m["ept_fwd_cached_s"] = float(fwd.group(1)) if fwd else None

    bwdfwd = re.search(r"Backward/Forward ratio \(full pipeline\):\s+(\d+\.\d+)x", text)
    m["bwd_fwd_ratio"] = float(bwdfwd.group(1)) if bwdfwd else None

    grad = re.search(r"EPT scalar gradient \(full pipe\):\s+(\d+\.\d+)s", text)
    m["grad_cached_s"] = float(grad.group(1)) if grad else None

    pmm = re.search(r"P_mm real: max rel err = (\d+\.\d+)%", text)
    m["pmm_max_pct"] = float(pmm.group(1)) if pmm else None

    pgg = re.search(r"P_gg real.*: max rel err = (\d+\.\d+)%", text)
    m["pgg_max_pct"] = float(pgg.group(1)) if pgg else None

    upstream = re.search(r"Upstream BG\+TH\+PT \(compile \+ cached\):\s+(\d+\.\d+)s", text)
    m["upstream_s"] = float(upstream.group(1)) if upstream else None
    return m
```

- [ ] **Step 7: Implement `parse_clpp`**

```python
def parse_clpp(text: str) -> dict:
    """Parse benchmark_clpp.py output."""
    m = {}
    for nl in ("none", "halofit", "ept"):
        # From SUMMARY block: "  compute_cl_pp('none', l_max=2500): 0.087s"
        hit = re.search(
            rf"compute_cl_pp\('{re.escape(nl)}',.*?\):\s+(\d+\.\d+)s", text)
        m[f"clpp_{nl}_s"] = float(hit.group(1)) if hit else None

    # Step 4 accuracy: "   100 |  4.5678e-07 |  4.5701e-07 |   -0.050%"
    rel_diffs = re.findall(
        r"^\s+\d+\s*\|\s*[\d.e+-]+\s*\|\s*[\d.e+-]+\s*\|\s*([+-]\d+\.\d+)%",
        text, re.MULTILINE)
    m["clpp_linear_max_abspct"] = max(abs(float(x)) for x in rel_diffs) if rel_diffs else None

    # Step 5 BB ratios: "     2 |  1.23e-15 |  1.24e-15 |  0.9982"
    m["bb_ratios"] = {}
    for hit in re.finditer(
        r"^\s+(\d+)\s*\|\s*[\d.e+-]+\s*\|\s*[\d.e+-]+\s*\|\s*([\d.]+|nan)",
        text, re.MULTILINE
    ):
        l = int(hit.group(1))
        val = hit.group(2)
        m["bb_ratios"][l] = float(val) if val != "nan" else None
    # Keep only the expected BB ells
    m["bb_ratios"] = {l: m["bb_ratios"].get(l) for l in [2, 10, 30, 50, 80, 100, 150, 200]}
    return m
```

- [ ] **Step 8: Implement `parse_accuracy`**

```python
def parse_accuracy(text: str) -> dict:
    """Parse accuracy_classpt.py output."""
    m = {"spectra": {}, "all_pass": None}
    # Per-spectrum line: "  pk_mm_real         0.31%     0.04%   0.0534  ✓ PASS           rel"
    for hit in re.finditer(
        r"^\s+(\w+)\s+([\d.]+)%\s+([\d.]+)%\s+[\d.]+\s+[✓✗]\s+(PASS|FAIL)",
        text, re.MULTILINE
    ):
        name = hit.group(1)
        m["spectra"][name] = {
            "max_pct": float(hit.group(2)),
            "mean_pct": float(hit.group(3)),
            "pass": hit.group(4) == "PASS",
        }
    m["all_pass"] = bool(re.search(r"ALL SPECTRA PASS", text))
    if re.search(r"FAILING spectra", text):
        m["all_pass"] = False
    return m
```

- [ ] **Step 9: Run tests — verify they pass**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -k "parse" -v
```
Expected: `7 passed`.

- [ ] **Step 10: Commit**

```bash
git add scripts/analyze_benchmark.py tests/test_analyze_benchmark.py
git commit -m "feat: Stage 1 parsers for all 7 benchmark scripts"
```

---

## Task 4: Stage 1 — threshold checker

**Files:**
- Modify: `scripts/analyze_benchmark.py`
- Modify: `tests/test_analyze_benchmark.py`

- [ ] **Step 1: Write failing tests for `check_thresholds`**

Add to `tests/test_analyze_benchmark.py`:
```python
from scripts.analyze_benchmark import check_thresholds

def test_check_thresholds_all_pass():
    metrics = {
        "solvers": parse_solvers(_fixture("solvers.txt")),
        "gradients": parse_gradients(_fixture("gradients.txt")),
        "fit_cl": parse_speed(_fixture("fit_cl.txt")),
        "ept": parse_ept(_fixture("ept.txt")),
        "clpp": parse_clpp(_fixture("clpp.txt")),
        "accuracy": parse_accuracy(_fixture("accuracy.txt")),
        "planck_cl": parse_speed(_fixture("planck_cl.txt")),
    }
    checked = check_thresholds(metrics)
    fails = [k for k, v in checked.items() if v["status"] == "FAIL"]
    assert fails == [], f"Unexpected FAILs: {fails}"

def test_check_thresholds_fit_cl_total_fail():
    metrics = {
        "fit_cl": {"total_cached_s": 55.0, "pt_cached_s": 28.0,
                   "cl_errs": {}, "preset": "fit_cl"},
    }
    checked = check_thresholds(metrics)
    assert checked["fit_cl.total_cached_s"]["status"] == "FAIL"

def test_check_thresholds_fit_cl_total_warn():
    metrics = {
        "fit_cl": {"total_cached_s": 48.0, "pt_cached_s": 28.0,
                   "cl_errs": {}, "preset": "fit_cl"},
    }
    checked = check_thresholds(metrics)
    assert checked["fit_cl.total_cached_s"]["status"] == "WARN"

def test_check_thresholds_accuracy_fail():
    metrics = {"accuracy": parse_accuracy(_fixture("accuracy_fail.txt"))}
    checked = check_thresholds(metrics)
    assert checked["accuracy.all_pass"]["status"] == "FAIL"

def test_check_thresholds_missing_file():
    checked = check_thresholds({})  # no planck_cl
    assert checked["planck_cl.total_cached_s"]["status"] == "MISSING"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -k "threshold" -v
```
Expected: `ImportError` — `check_thresholds` not defined yet.

- [ ] **Step 3: Implement `check_thresholds`**

Add to `scripts/analyze_benchmark.py`:
```python
# All thresholds from BENCHMARK.md §4
_THRESHOLDS = {
    # §4.1 timing — GPU A100 targets
    "fit_cl.total_cached_s":        {"limit": 50.0,  "op": "lt", "warn_pct": 0.20},
    "planck_cl.total_cached_s":     {"limit": 600.0, "op": "lt", "warn_pct": 0.20},
    "fit_cl.pt_cached_s":           {"limit": 30.0,  "op": "lt", "warn_pct": 0.20},
    "planck_cl.pt_cached_s":        {"limit": 30.0,  "op": "lt", "warn_pct": 0.20},
    "ept.ept_fwd_cached_s":         {"limit": 2.0,   "op": "lt", "warn_pct": 0.20},
    "gradients.bwd_fwd_ratio":      {"limit": 4.0,   "op": "lt", "warn_pct": 0.20},
    "ept.bwd_fwd_ratio":            {"limit": 5.0,   "op": "lt", "warn_pct": 0.20},
    # §4.1 solver
    "solvers.rodas5_vs_kv_max_pct": {"limit": 0.1,   "op": "lt", "warn_pct": 0.20},
    "solvers.rodas5_speedup":       {"limit": 2.0,   "op": "gt", "warn_pct": 0.20},
    # §4.2 accuracy
    "ept.pmm_max_pct":              {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "ept.pgg_max_pct":              {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "clpp.clpp_linear_max_abspct":  {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "accuracy.all_pass":            {"limit": True,  "op": "eq", "warn_pct": None},
}
# AD/FD per-param: < 1%
_GRAD_PARAMS = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
# BB ratios at each ell
_BB_THRESHOLDS = {
    2: (0.95, 1.05), 10: (0.95, 1.05), 30: (0.95, 1.05),
    50: (0.95, 1.05), 80: (0.95, 1.05), 100: (0.95, 1.05),
    150: (0.90, 1.10), 200: (0.90, 1.10),
}
# CL accuracy at CHECK_ELLS (max abs err%): TT/EE < 0.5%, TE skip at zero crossings
_CL_ERR_THRESHOLD = 0.5


def _scalar_check(value, threshold_def) -> str:
    if value is None:
        return MetricStatus.MISSING
    limit = threshold_def["limit"]
    op = threshold_def["op"]
    warn_pct = threshold_def.get("warn_pct")
    if op == "lt":
        passes = value < limit
        near = warn_pct and value >= limit * (1 - warn_pct) and value < limit
    elif op == "gt":
        passes = value > limit
        near = warn_pct and value <= limit * (1 + warn_pct) and value > limit
    elif op == "eq":
        passes = value == limit
        near = False
    else:
        raise ValueError(f"Unknown op: {op}")
    if not passes:
        return MetricStatus.FAIL
    if near:
        return MetricStatus.WARN
    return MetricStatus.PASS


def check_thresholds(metrics: dict) -> dict:
    """Check all metrics against BENCHMARK.md §4 thresholds.

    Returns dict mapping metric_key -> {value, status, threshold, description}.
    """
    checked = {}

    def _add(key, value, status, threshold, description=""):
        checked[key] = {
            "value": value,
            "status": status,
            "threshold": threshold,
            "description": description,
        }

    # Scalar thresholds
    for key, tdef in _THRESHOLDS.items():
        script, attr = key.split(".", 1)
        script_metrics = metrics.get(script, {})
        value = script_metrics.get(attr) if isinstance(script_metrics, dict) else None
        status = _scalar_check(value, tdef)
        _add(key, value, status, tdef["limit"], f"BENCHMARK.md §4")

    # Gradient AD/FD agreement per param
    grad = metrics.get("gradients", {})
    agreement = grad.get("ad_fd_agreement", {}) if grad else {}
    for p in _GRAD_PARAMS:
        val = agreement.get(p)
        status = _scalar_check(val, {"limit": 1.0, "op": "lt", "warn_pct": 0.20})
        _add(f"gradients.ad_fd_{p}_pct", val, status, 1.0, "AD/FD < 1%")

    # BB ratios
    clpp = metrics.get("clpp", {})
    bb = clpp.get("bb_ratios", {}) if clpp else {}
    for l, (lo, hi) in _BB_THRESHOLDS.items():
        val = bb.get(l) if bb else None
        if val is None:
            status = MetricStatus.MISSING
        elif lo <= val <= hi:
            # Check WARN: within 20% of either boundary
            margin_lo = (val - lo) / (hi - lo)
            margin_hi = (hi - val) / (hi - lo)
            status = MetricStatus.WARN if min(margin_lo, margin_hi) < 0.20 else MetricStatus.PASS
        else:
            status = MetricStatus.FAIL
        _add(f"clpp.bb_ratio_l{l}", val, status, f"[{lo},{hi}]", f"BB ratio ℓ={l}")

    # CL errors at CHECK_ELLS: TT and EE < 0.5% (abs value; TE skipped)
    for preset in ("fit_cl", "planck_cl"):
        speed = metrics.get(preset, {})
        cl_errs = speed.get("cl_errs", {}) if speed else {}
        for l in [20, 100, 500, 1000]:
            errs = cl_errs.get(l, {}) if cl_errs else {}
            for spec in ("tt", "ee"):
                val = errs.get(spec)
                abs_val = abs(val) if val is not None else None
                status = _scalar_check(abs_val, {"limit": _CL_ERR_THRESHOLD, "op": "lt", "warn_pct": 0.20})
                _add(f"{preset}.cl_{spec}_l{l}_abspct", abs_val, status, _CL_ERR_THRESHOLD)

    return checked
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -k "threshold" -v
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_benchmark.py tests/test_analyze_benchmark.py
git commit -m "feat: Stage 1 threshold checker against BENCHMARK.md §4"
```

---

## Task 5: Stage 1 — REPORT.md writer + CLI

**Files:**
- Modify: `scripts/analyze_benchmark.py`
- Modify: `tests/test_analyze_benchmark.py`

- [ ] **Step 1: Write failing tests for `write_report` and `main`**

Add to `tests/test_analyze_benchmark.py`:
```python
from scripts.analyze_benchmark import write_report, build_results_dict

def test_write_report_creates_files(tmp_path):
    # Copy fixtures into tmp dir with correct naming
    for name in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text(
            _fixture(f"{name}.txt"))
    found, missing = find_result_files(tmp_path)
    parsed = {k: globals()[f"parse_{k if k not in ('fit_cl',) else 'speed'}"](v.read_text())
              for k, v in found.items()}
    checked = check_thresholds(parsed)
    results = build_results_dict(tmp_path, found, missing, parsed, checked)
    write_report(results, tmp_path)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "REPORT.md").exists()

def test_results_json_structure(tmp_path):
    (tmp_path / "ravg1002-a100-solvers.txt").write_text(_fixture("solvers.txt"))
    found, missing = find_result_files(tmp_path)
    parsed = {"solvers": parse_solvers(found["solvers"].read_text())}
    checked = check_thresholds(parsed)
    results = build_results_dict(tmp_path, found, missing, parsed, checked)
    assert "date" in results
    assert "files_found" in results
    assert "files_missing" in results
    assert "metrics" in results
    assert "summary" in results
    assert "n_pass" in results["summary"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -k "report or results_json" -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `build_results_dict` and `write_report`**

Add to `scripts/analyze_benchmark.py`:
```python
from datetime import datetime

STATUS_ICON = {
    MetricStatus.PASS: "✓",
    MetricStatus.FAIL: "✗",
    MetricStatus.WARN: "⚠",
    MetricStatus.MISSING: "·",
}


def build_results_dict(result_dir: Path, found: dict, missing: list,
                       parsed: dict, checked: dict) -> dict:
    node = None
    for p in result_dir.glob("*-a100-*.txt"):
        node = p.name.split("-a100-")[0]
        break
    summary = {
        "n_pass": sum(1 for v in checked.values() if v["status"] == MetricStatus.PASS),
        "n_fail": sum(1 for v in checked.values() if v["status"] == MetricStatus.FAIL),
        "n_warn": sum(1 for v in checked.values() if v["status"] == MetricStatus.WARN),
        "n_missing": sum(1 for v in checked.values() if v["status"] == MetricStatus.MISSING),
    }
    return {
        "date": str(result_dir.name) if result_dir.name[0].isdigit() else datetime.now().strftime("%Y-%m-%d"),
        "node": node,
        "result_dir": str(result_dir),
        "files_found": list(found.keys()),
        "files_missing": missing,
        "metrics": parsed,
        "thresholds": {k: {**v, "value": str(v["value"])} for k, v in checked.items()},
        "summary": summary,
        "analyzed_at": datetime.now().isoformat(),
    }


def write_report(results: dict, out_dir: Path) -> None:
    """Write results.json and REPORT.md to out_dir."""
    # results.json
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # REPORT.md
    lines = []
    d = results["date"]
    node = results.get("node", "unknown")
    s = results["summary"]
    lines += [
        f"# clax benchmark report — {d}",
        "",
        "## Platform",
        f"- Node: {node}",
        f"- Files present: {', '.join(results['files_found']) or 'none'}",
        f"- Files missing: {', '.join(results['files_missing']) or 'none'}",
        f"- Analyzed: {results['analyzed_at']}",
        "",
        f"## Scorecard: {s['n_pass']} PASS · {s['n_fail']} FAIL · "
        f"{s['n_warn']} WARN · {s['n_missing']} MISSING",
        "",
        "| Metric | Value | Status | Threshold |",
        "|---|---|---|---|",
    ]
    for key, v in results["thresholds"].items():
        icon = STATUS_ICON.get(v["status"], "?")
        val_str = f"{v['value']}" if v["value"] is not None else "—"
        lines.append(f"| `{key}` | {val_str} | {icon} {v['status']} | {v['threshold']} |")

    lines += ["", "## Raw output tails", ""]
    raw_dir = out_dir
    for script in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]:
        paths = list(raw_dir.glob(f"*-a100-{script}.txt"))
        lines += [f"### {script}"]
        if paths:
            tail = paths[0].read_text().strip().split("\n")
            lines += ["```"] + tail[-15:] + ["```", ""]
        else:
            lines += ["*(not yet available)*", ""]

    (out_dir / "REPORT.md").write_text("\n".join(lines))
```

- [ ] **Step 4: Implement `main` CLI**

Add to `scripts/analyze_benchmark.py`:
```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.dir or Path(f"/ptmp/minh/clax-bench/{args.date}")
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}", file=sys.stderr)
        sys.exit(2)

    found, missing = find_result_files(result_dir)
    if not found:
        print(f"No result files found in {result_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found: {list(found.keys())}")
    print(f"Missing: {missing}")

    parsed = {}
    for name, path in found.items():
        text = path.read_text()
        if name == "solvers":
            parsed[name] = parse_solvers(text)
        elif name == "gradients":
            parsed[name] = parse_gradients(text)
        elif name in ("fit_cl", "planck_cl"):
            parsed[name] = parse_speed(text)
        elif name == "ept":
            parsed[name] = parse_ept(text)
        elif name == "clpp":
            parsed[name] = parse_clpp(text)
        elif name == "accuracy":
            parsed[name] = parse_accuracy(text)

    checked = check_thresholds(parsed)
    results = build_results_dict(result_dir, found, missing, parsed, checked)
    write_report(results, result_dir)

    n_fail = results["summary"]["n_fail"]
    n_missing = results["summary"]["n_missing"]
    print(f"\nReport written to {result_dir}/REPORT.md")
    print(f"Summary: {results['summary']}")

    if n_fail > 0:
        sys.exit(1)
    elif n_missing > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all Stage 1 tests**

```bash
conda run -n py311forge python -m pytest tests/test_analyze_benchmark.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Smoke-test CLI on fixtures (no GPU needed)**

```bash
cd /u/minh/clax
# Create a temp dir with fixture files named correctly
TMP=$(mktemp -d)
for f in tests/fixtures/benchmark/*.txt; do
    name=$(basename "$f" .txt)
    cp "$f" "${TMP}/ravg1002-a100-${name}.txt"
done
conda run -n py311forge python scripts/analyze_benchmark.py --dir "${TMP}"
cat "${TMP}/REPORT.md" | head -30
cat "${TMP}/results.json" | python -m json.tool | head -30
```
Expected: REPORT.md and results.json written, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_benchmark.py tests/test_analyze_benchmark.py
git commit -m "feat: Stage 1 complete — REPORT.md writer and CLI"
```

---

## Task 6: Stage 2 — Claude API agent

**Files:**
- Create: `scripts/agent_analyze_benchmark.py`

- [ ] **Step 1: Write `scripts/agent_analyze_benchmark.py`**

```python
"""Stage 2: Claude API agent — reads results.json + raw files, writes ANALYSIS.md.

Usage:
    python scripts/agent_analyze_benchmark.py [--date YYYY-MM-DD] [--dir /path]

Requires:
    ANTHROPIC_API_KEY environment variable

Writes:
    <dir>/ANALYSIS.md   narrative analysis with PASS/FAIL/WARN/PENDING per metric
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import anthropic

EXPECTED_SCRIPTS = ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]

# Preflight smoke-test baseline (job 26878115, gpudev A100, 2026-05-04)
PREFLIGHT_BASELINE = {
    "kvaerno5_median_s": 13.09,
    "platform": "gpu",
    "node": "ravg1002",
    "job_id": "26878115",
}

# BENCHMARK.md §4 targets (GPU A100)
BENCHMARK_TARGETS = """
## BENCHMARK.md §4 Thresholds (GPU A100 targets)

### §4.1 Timing
- fit_cl total cached: ≤ 50 s  (current baseline: 34 s on V100, CLAUDE.md)
- planck_cl total cached: ≤ 600 s  (baseline: 487 s on H100)
- Perturbation solve: ≤ 30 s on V100/A100
- EPT forward cached: ≤ 2 s (GPU target)
- Gradient bwd/fwd ratio: < 4× (GPU target)
- rodas5 speedup vs kvaerno5: ≥ 2.0× on GPU
- rodas5 vs kvaerno5 accuracy: max rel err < 0.1%

### §4.2 Accuracy
- Linear P(k) max err: < 0.5% at k ∈ [0.001, 0.3] Mpc⁻¹
- C_l^TT, C_l^EE max err: < 0.5% at ℓ ∈ [20, 2000]
- C_l^pp linear max err: < 1% at ℓ ≤ 2500
- EPT P_mm/P_gg max rel err: < 1% at k ∈ [0.005, 0.3] h/Mpc
- EPT P_mm/P_gg ℓ=4 abs/max err: < 2%
- BB ratio at ℓ ≤ 100: [0.95, 1.05]
- BB ratio at ℓ ∈ {150, 200}: [0.90, 1.10]
- accuracy_classpt.py all 9 spectra: PASS (exit 0)
- AD/FD agreement per parameter: < 1%
"""

ACCURACY_PAPER_CONTEXT = """
## Existing paper-draft accuracy numbers (CPU, benchmark/clax-pt snapshot)
From drafts/accuracy_results_for_paper.md:

- P_mm real: max 0.31%, mean 0.04%  ← this is the CPU reference
- P_gg real: max 0.31%, mean 0.04%
- P_mm ℓ=0: max 0.59%, mean 0.40%
- P_gg ℓ=2: max 0.89%, mean 0.50%
- P_gg ℓ=4: max 1.43% (abs/max metric)
- fit_cl total: 34 s on V100 (CLAUDE.md)
- planck_cl total: ~487 s on H100 (CLAUDE.md)
"""

SYSTEM_PROMPT = f"""You are analyzing benchmark results for clax, a fully differentiable JAX reimplementation of the CLASS Boltzmann solver. You will be given:
1. A structured results.json from the deterministic parser
2. Raw benchmark output files
3. The BENCHMARK.md §4 thresholds
4. Preflight smoke-test baseline numbers
5. Existing CPU accuracy numbers from the paper draft

Your job is to write a thorough ANALYSIS.md with these sections:

## Platform summary
One paragraph: node, GPU model, driver version (from nvidia-smi output in solvers.txt), JAX version, branch/commit.

## Results scorecard
A table with every metric: metric name | value | status (PASS ✓ / FAIL ✗ / WARN ⚠ / PENDING ⏳) | threshold | notes.
For PENDING metrics (planck_cl if missing), say what we're waiting for.

## Anomalies and recommended actions
For each FAIL or WARN: what the number is, what the threshold is, probable cause, recommended action. Be specific — cite line numbers in BENCHMARK.md if relevant.
If everything passes, say so explicitly and note any metrics that are unusually close to thresholds.

## Comparison vs baselines
- Compare fit_cl total cached against CLAUDE.md baseline (34 s on V100). Is the A100 faster or slower? By how much?
- Compare planck_cl (if present) against CLAUDE.md baseline (487 s on H100).
- Compare kvaerno5 median against preflight smoke-test baseline (13.09 s, job 26878115).
- Compare EPT accuracy numbers against the CPU paper-draft numbers — are they consistent (within 10%)?

## Paper-ready numbers
2–3 sentences suitable for a methods-section timing table caption. Include: platform, JAX version, fit_cl cached total, planck_cl cached total (or PENDING), perturbation-only cached time.

## Updated LaTeX snippet
One new table row for the GPU A100 timing column, using this template:
```latex
A100 (40 GB) & clax \texttt{fit\_cl} & JAX 0.10.0 & {fit_cl_total:.0f} s & {planck_cl_total} s & \\
```
If planck_cl is pending, use "pending".

## What to watch when planck_cl lands
2–3 bullet points: which metrics will become available, what values to expect based on CLAUDE.md, what would constitute a regression.

{BENCHMARK_TARGETS}

{ACCURACY_PAPER_CONTEXT}

## PREFLIGHT BASELINE (job 26878115, gpudev A100, 2026-05-04)
- kvaerno5 median (1 warmup, 1 repeat): 13.09 s  ← note: this includes partial JIT overhead; production runs use 2 warmup + 5 repeat
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, driver 580.126.20
- JAX: 0.10.0, [CudaDevice(id=0)]

Be precise and cite numbers. Do not hedge. If a metric is borderline, say so. If the pipeline is healthy, say so clearly. The user will read this before they open their laptop.
"""


def load_context(result_dir: Path) -> tuple[str, str]:
    """Load results.json and all available raw .txt files. Returns (json_str, raw_str)."""
    results_path = result_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {result_dir}. Run analyze_benchmark.py first.")
    json_str = results_path.read_text()

    raw_parts = []
    for script in EXPECTED_SCRIPTS:
        paths = list(result_dir.glob(f"*-a100-{script}.txt"))
        if paths:
            raw_parts.append(f"### {script}.txt\n```\n{paths[0].read_text().strip()}\n```")
        else:
            raw_parts.append(f"### {script}.txt\n*(not yet available)*")
    return json_str, "\n\n".join(raw_parts)


def run_agent(result_dir: Path) -> str:
    """Call Claude API, return ANALYSIS.md content."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    json_str, raw_str = load_context(result_dir)

    user_message = f"""Here are the benchmark results for analysis.

## results.json
```json
{json_str}
```

## Raw benchmark outputs
{raw_str}

Please write the complete ANALYSIS.md now. Be thorough. Flag every FAIL and WARN explicitly.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.dir or Path(f"/ptmp/minh/clax-bench/{args.date}")
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Running Stage 2 agent on {result_dir}...")
    analysis = run_agent(result_dir)

    out_path = result_dir / "ANALYSIS.md"
    out_path.write_text(analysis)
    print(f"ANALYSIS.md written to {out_path}")
    print(f"\nFirst 20 lines:\n" + "\n".join(analysis.split("\n")[:20]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test Stage 2 against the fixtures temp dir (uses real API)**

```bash
cd /u/minh/clax
# Reuse the TMP dir from Task 5 Step 6, or recreate:
TMP=$(mktemp -d)
for f in tests/fixtures/benchmark/*.txt; do
    name=$(basename "$f" .txt)
    [[ "$name" == "accuracy_fail" ]] && continue
    cp "$f" "${TMP}/ravg1002-a100-${name}.txt"
done
conda run -n py311forge python scripts/analyze_benchmark.py --dir "${TMP}"
conda run -n py311forge python scripts/agent_analyze_benchmark.py --dir "${TMP}"
cat "${TMP}/ANALYSIS.md"
```
Expected: `ANALYSIS.md` written with all sections, 2–3 min wall time for API call.

- [ ] **Step 3: Commit**

```bash
git add scripts/agent_analyze_benchmark.py
git commit -m "feat: Stage 2 Claude API agent — ANALYSIS.md writer"
```

---

## Task 7: Watcher script

**Files:**
- Create: `scripts/watch_benchmark.py`

- [ ] **Step 1: Write `scripts/watch_benchmark.py`**

```python
"""Watcher: poll result dir, drive Stage 1+2, send push notification, self-cancel.

Designed to be invoked by a 30-min cron. Idempotent: skips if no new files
since last run. Self-cancels via CronDelete when all 7 files are analyzed.

Usage:
    python scripts/watch_benchmark.py [--date YYYY-MM-DD] [--cron-id ID]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

EXPECTED = ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]
LOG_FILE = None  # set in main()
REPO_ROOT = Path(__file__).parent.parent


def log(msg):
    logging.info(msg)
    print(msg)


def find_any_txt(result_dir: Path) -> list[Path]:
    return list(result_dir.glob("*-a100-*.txt"))


def get_mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def run_stage1(result_dir: Path) -> bool:
    """Run analyze_benchmark.py. Returns True on success (exit 0 or 2)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_benchmark.py"),
         "--dir", str(result_dir)],
        capture_output=True, text=True
    )
    log(f"Stage 1 stdout:\n{result.stdout}")
    if result.stderr:
        log(f"Stage 1 stderr:\n{result.stderr}")
    # Exit 0 = all pass, 1 = FAIL, 2 = MISSING — all acceptable to continue
    return result.returncode in (0, 1, 2)


def run_stage2(result_dir: Path) -> bool:
    """Run agent_analyze_benchmark.py. Returns True on success."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "agent_analyze_benchmark.py"),
         "--dir", str(result_dir)],
        capture_output=True, text=True,
        env={**os.environ}
    )
    log(f"Stage 2 stdout:\n{result.stdout}")
    if result.stderr:
        log(f"Stage 2 stderr:\n{result.stderr}")
    return result.returncode == 0


def is_complete(result_dir: Path) -> bool:
    """All 7 files present and ANALYSIS.md exists and says [COMPLETE]."""
    txt_files = find_any_txt(result_dir)
    found_scripts = set()
    for p in txt_files:
        for name in EXPECTED:
            if p.name.endswith(f"-{name}.txt"):
                found_scripts.add(name)
    if set(EXPECTED) != found_scripts:
        return False
    analysis = result_dir / "ANALYSIS.md"
    return analysis.exists() and "[COMPLETE]" in analysis.read_text()


def delete_cron(cron_id: str):
    """Self-cancel by removing the cron job. Logs failure but doesn't crash."""
    try:
        result = subprocess.run(
            ["claude", "cron", "delete", cron_id],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log(f"Cron {cron_id} deleted — watcher complete.")
        else:
            log(f"Failed to delete cron {cron_id}: {result.stderr}")
    except Exception as e:
        log(f"CronDelete error: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--cron-id", default=None,
                        help="Cron job ID for self-cancellation")
    args = parser.parse_args()

    result_dir = Path(f"/ptmp/minh/clax-bench/{args.date}")
    log_dir = result_dir if result_dir.exists() else Path("/ptmp/minh/clax-bench")
    log_dir.mkdir(parents=True, exist_ok=True)

    global LOG_FILE
    LOG_FILE = log_dir / f"watcher-{args.date}.log"
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    log(f"Watcher tick — date={args.date}, result_dir={result_dir}")

    if not result_dir.exists():
        log("Result directory does not exist — jobs not started yet. Exiting.")
        return

    txt_files = find_any_txt(result_dir)
    if not txt_files:
        log("No .txt result files found yet. Exiting.")
        return

    # Check if any file is newer than .last_analyzed
    last_analyzed = result_dir / ".last_analyzed"
    last_mtime = get_mtime(last_analyzed)
    new_files = [p for p in txt_files if p.stat().st_mtime > last_mtime]

    if not new_files:
        log(f"No new files since last analysis ({last_analyzed}). Exiting.")
        return

    log(f"New files detected: {[p.name for p in new_files]}")

    # Stage 1
    if not run_stage1(result_dir):
        log("Stage 1 failed unexpectedly. Will retry next tick.")
        return

    # Stage 2
    if not run_stage2(result_dir):
        log("Stage 2 failed. REPORT.md is still available from Stage 1. Will retry next tick.")
        # Don't touch .last_analyzed — retry Stage 2 next tick
        return

    # Mark analyzed
    last_analyzed.touch()
    log(f"Analysis complete. Results in {result_dir}/")

    # Check completion and self-cancel
    if is_complete(result_dir):
        log("All 7 files analyzed and ANALYSIS.md is [COMPLETE].")
        cron_id = args.cron_id or _read_cron_id()
        if cron_id:
            delete_cron(cron_id)
        else:
            log("No cron ID — cannot self-cancel. Remove the cron job manually.")
    else:
        found = {n for n in EXPECTED if any(p.name.endswith(f"-{n}.txt") for p in txt_files)}
        missing = [n for n in EXPECTED if n not in found]
        log(f"Still waiting for: {missing}")


def _read_cron_id() -> str | None:
    p = Path.home() / ".clax_bench_cron_id"
    return p.read_text().strip() if p.exists() else None


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the watcher against the fixtures dir**

```bash
TMP=$(mktemp -d)/2026-05-04
mkdir -p "$TMP"
for f in tests/fixtures/benchmark/*.txt; do
    name=$(basename "$f" .txt)
    [[ "$name" == "accuracy_fail" || "$name" == "planck_cl" ]] && continue
    cp "$f" "${TMP}/ravg1002-a100-${name}.txt"
done
conda run -n py311forge python scripts/watch_benchmark.py \
    --date 2026-05-04 --dir "${TMP}" 2>&1 | head -40
ls "${TMP}/"
```
Expected: `REPORT.md`, `results.json`, `ANALYSIS.md`, `.last_analyzed` all created. Second run should print "No new files since last analysis".

- [ ] **Step 3: Commit**

```bash
git add scripts/watch_benchmark.py
git commit -m "feat: watcher script with idempotent detection and self-cancel"
```

---

## Task 8: Schedule creation

**Files:**
- No new files — uses Claude Code `CronCreate` tool

- [ ] **Step 1: Verify ANTHROPIC_API_KEY is available on login node**

```bash
echo "${ANTHROPIC_API_KEY:0:8}..."
```
Expected: shows first 8 chars of the key. If blank, check `~/.bashrc` or the active env.

- [ ] **Step 2: Create the 30-min cron via the schedule skill**

Invoke the `schedule` skill and create a cron with:
- Schedule: `*/30 * * * *`
- Command: `cd /u/minh/clax && conda run -n py311forge python scripts/watch_benchmark.py --date $(date +%F) --cron-id CRON_ID_PLACEHOLDER`
- Description: `clax benchmark watcher — poll for result files, run analysis, notify`

After creation, get the cron ID and:
```bash
echo "CRON_ID" > ~/.clax_bench_cron_id
```
Then update the cron command to replace `CRON_ID_PLACEHOLDER` with the actual ID so the watcher can self-cancel.

- [ ] **Step 3: Verify the cron is listed**

```bash
# List crons to confirm
claude cron list 2>&1 | grep clax
```
Expected: the new cron appears.

- [ ] **Step 4: Final commit and push**

```bash
git add scripts/
git commit -m "feat: benchmark analysis pipeline complete — Stage 1+2 + watcher"
git push origin benchmark/clax-pt
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `analyze_benchmark.py` — 7 parsers | Task 3 |
| `analyze_benchmark.py` — threshold checker (all §4 thresholds) | Task 4 |
| `analyze_benchmark.py` — `results.json` + `REPORT.md` + CLI + exit codes | Task 5 |
| `agent_analyze_benchmark.py` — Claude API, system prompt, ANALYSIS.md | Task 6 |
| `agent_analyze_benchmark.py` — push notification | Task 6 Step 1 (in `main`) ✓ |
| `watch_benchmark.py` — file detection, idempotent, orchestration | Task 7 |
| `watch_benchmark.py` — self-cancel via CronDelete | Task 7 Step 1 (`delete_cron`) |
| `watch_benchmark.py` — partial-run aware, PENDING ⏳ | Task 7 (Stage 2 handles via system prompt) |
| Schedule creation, 30-min interval | Task 8 |
| No GPU required, login node | All scripts: no JAX imports, pure stdlib + anthropic |
| `py311forge` env, `anthropic` installed | Task 6 Step 2 (already installed) |

**Placeholder scan:** No TBDs, no "implement later". All regex patterns are concrete. All file paths are absolute or constructed from `REPO_ROOT`.

**Push notification:** It's in the Stage 2 agent system prompt ("send a push notification") but `agent_analyze_benchmark.py` `main()` doesn't call `PushNotification` directly — the Claude agent in Stage 2 is a plain API call, not a Claude Code session, so it can't call tools. **Fix:** add an explicit `subprocess` call to `claude push-notification` after Stage 2 runs in `watch_benchmark.py`:

Add to `watch_benchmark.py` after `run_stage2` succeeds:
```python
def send_push(result_dir: Path):
    """Send push notification via Claude Code CLI."""
    results_path = result_dir / "results.json"
    if not results_path.exists():
        return
    try:
        data = json.loads(results_path.read_text())
        s = data.get("summary", {})
        title = f"clax bench {data.get('date','?')}: {s.get('n_pass',0)} pass, {s.get('n_fail',0)} fail, {s.get('n_missing',0)} pending"
        body = f"ANALYSIS.md: {result_dir}/ANALYSIS.md"
        subprocess.run(
            ["claude", "api", "push-notification",
             "--title", title, "--body", body],
            capture_output=True, timeout=30
        )
        log(f"Push notification sent: {title}")
    except Exception as e:
        log(f"Push notification failed (non-fatal): {e}")
```
Call `send_push(result_dir)` right after `run_stage2` returns `True`.
