# Benchmark Analysis Pipeline — Design Spec
**Date:** 2026-05-04
**Branch:** `benchmark/clax-pt`
**Author:** Claude Sonnet 4.6 (session with Minh)

---

## 1. Problem statement

Two SLURM jobs (26878941 `clax-bench-fast`, 26878942 `clax-bench-planck`) will deposit 7 plaintext result files into `/ptmp/minh/clax-bench/<DATE>/`. The partial `REPORT.md` written by the sbatch scripts captures raw tail lines but does not parse metrics, check thresholds, or flag anomalies. The goal is a pipeline that:

- Fires automatically when result files land (no manual babysitting required)
- Parses every key metric from each script's output
- Checks each metric against the BENCHMARK.md §4 thresholds
- Produces a complete, human-readable `REPORT.md` and a deeper `ANALYSIS.md`
- Sends a push notification so results are reviewed before the user reaches their laptop
- Updates the paper-draft LaTeX table in `drafts/accuracy_results_for_paper.md` with GPU timing numbers

---

## 2. Architecture

```
/ptmp/minh/clax-bench/<DATE>/
  <node>-a100-solvers.txt          ┐
  <node>-a100-gradients.txt        │  raw benchmark outputs (sbatch jobs)
  <node>-a100-fit_cl.txt           │
  <node>-a100-ept.txt              │
  <node>-a100-clpp.txt             │
  <node>-a100-accuracy.txt         │
  <node>-a100-planck_cl.txt        ┘  (arrives with second job)
  results.json                        ← Stage 1 output
  REPORT.md                           ← Stage 1 output (overwrites sbatch partial)
  ANALYSIS.md                         ← Stage 2 output
  .last_analyzed                      ← watcher state file (mtime tracking)

scripts/analyze_benchmark.py          ← Stage 1: deterministic parser + threshold checker
scripts/agent_analyze_benchmark.py    ← Stage 2: Claude API reasoning agent
scripts/watch_benchmark.py            ← Watcher: detects new files, orchestrates stages
```

The scheduler fires `watch_benchmark.py` every 30 minutes via CronCreate (`*/30 * * * *`).

---

## 3. Stage 1 — `scripts/analyze_benchmark.py`

### 3.1 Invocation

```bash
python scripts/analyze_benchmark.py [--date YYYY-MM-DD] [--dir /ptmp/minh/clax-bench/DATE]
```

Defaults: `--date` = today (`date +%F`). Globs `<dir>/*-a100-*.txt` to find the result files.

### 3.2 Per-script parsers

Each parser is a function `parse_<name>(text: str) -> dict` that extracts metrics via regex from the script's SUMMARY block and structured print lines. Returns `None` for a metric if the line is absent (file incomplete or script crashed).

| File | Key metrics extracted |
|---|---|
| `solvers.txt` | `kvaerno5_median_s`, `rodas5_median_s`, `rodas5_speedup`, `rosenbrock_median_s`, `rodas5_vs_kv_max_pct`, `platform` |
| `gradients.txt` | `fwd_cached_s`, `ad_median_s`, `fd_median_s`, `bwd_fwd_ratio`, per-param `ad_fd_agreement_pct` for `{h, omega_b, omega_cdm, ln10A_s, n_s}`, `ad_fd_speedup` |
| `fit_cl.txt` | `bg_cached_s`, `th_cached_s`, `pt_cached_s`, `hr_cached_s`, `total_cached_s`, TT/EE/TE err at `l ∈ {20, 100, 500, 1000}` |
| `ept.txt` | `upstream_compile_s`, `ept_fwd_cached_s`, `multi_z_total_s`, `grad_cached_s`, `bwd_fwd_ratio`, `pmm_max_pct`, `pmm_mean_pct`, `pgg_max_pct`, `pgg_mean_pct` |
| `clpp.txt` | `upstream_s`, `clpp_none_s`, `clpp_halofit_s`, `clpp_ept_s`, `clpp_linear_max_pct` at spot-check ℓ values, `bb_ratios` at `l ∈ {2,10,30,50,80,100,150,200}` |
| `accuracy.txt` | per-spectrum `{name: {max_pct, mean_pct, pass}}` for all 9 CLASS-PT spectra, `all_pass` bool |
| `planck_cl.txt` | same shape as `fit_cl.txt` but `preset=planck_cl`; absent → all values `None` |

### 3.3 Threshold checks

All thresholds come directly from BENCHMARK.md §4. Each metric gets a status:

- `PASS` — within threshold
- `FAIL` — exceeds threshold
- `WARN` — within 20% of threshold (approaching limit)
- `MISSING` — file absent or metric line not found

| Metric | Threshold | Source |
|---|---|---|
| `fit_cl` total cached | ≤ 50 s | §4.1 |
| `planck_cl` total cached | ≤ 600 s | §4.1 |
| Perturbation solve only (`pt_cached_s`) | ≤ 30 s | §4.1 |
| EPT forward cached | ≤ 2 s (GPU target) | §4.1 |
| C_l^pp linear max err | < 1% at ℓ ≤ 2500 | §4.2 |
| Gradient bwd/fwd ratio | < 4× (GPU target) | §4.1 |
| AD/FD agreement per param | < 1% | §4.2 (gradient tests) |
| rodas5 vs kvaerno5 max rel err | < 0.1% | §3.1 |
| rodas5 speedup | ≥ 2.0× (GPU target) | §3.1 |
| Linear P(k) max err | < 0.5% | §4.2 |
| C_l^TT max err, ℓ ∈ [20,2000] | < 0.5% | §4.2 |
| C_l^EE max err, ℓ ∈ [20,2000] | < 0.5% | §4.2 |
| EPT P_mm/P_gg max rel err | < 1% | §4.2 |
| EPT P_mm/P_gg ℓ=4 abs/max | < 2% | §4.2 |
| BB ratio at ℓ ≤ 100 | [0.95, 1.05] | §4.2 |
| BB ratio at ℓ ∈ {150, 200} | [0.90, 1.10] | §4.2 |
| accuracy_classpt all 9 spectra | all PASS (exit 0) | §3.5 |

### 3.4 Outputs

**`results.json`** — structured dict:
```json
{
  "date": "2026-05-04",
  "node": "ravg1002",
  "files_found": ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy"],
  "files_missing": ["planck_cl"],
  "metrics": { "<script>": { "<metric>": { "value": ..., "status": "PASS|FAIL|WARN|MISSING", "threshold": ... } } },
  "summary": { "n_pass": 18, "n_fail": 0, "n_warn": 2, "n_missing": 4 },
  "analyzed_at": "2026-05-04T18:30:00"
}
```

**`REPORT.md`** — human-readable tables, one section per script, pass/fail badges, then a summary scorecard. Overwrites the partial sbatch-generated file.

### 3.5 CLI exit code

Exits `0` if all present metrics PASS, `1` if any FAIL, `2` if any MISSING (partial run).

---

## 4. Stage 2 — `scripts/agent_analyze_benchmark.py`

### 4.1 Invocation

```bash
python scripts/agent_analyze_benchmark.py [--date YYYY-MM-DD] [--dir /ptmp/minh/clax-bench/DATE]
```

Requires `ANTHROPIC_API_KEY` in environment.

### 4.2 Model and context

- Model: `claude-sonnet-4-6`
- Prompt caching enabled (system prompt + results.json + raw .txt files as cached prefix)
- Max tokens: 4096 for `ANALYSIS.md` output

### 4.3 System prompt scope

The agent is given:

1. `results.json` (Stage 1 output)
2. All available raw `.txt` files (full text)
3. The BENCHMARK.md §4 thresholds table verbatim
4. The CLAUDE.md performance baselines (fit_cl 34s V100, planck_cl 487s H100)
5. The preflight smoke-test numbers (kvaerno5 13.09s on this A100, job 26878115)
6. The existing `drafts/accuracy_results_for_paper.md` LaTeX table

The agent is instructed to reason over:
- Every FAIL and WARN with a probable cause and recommended action
- Whether GPU timings represent improvement or regression vs CLAUDE.md baselines and the preflight
- Whether EPT/accuracy numbers are consistent with `drafts/accuracy_results_for_paper.md` (same physics, different platform)
- What is still PENDING and what to watch for when `planck_cl.txt` arrives
- Paper-ready narrative: 2–3 sentences suitable for a methods-section timing table caption
- Updated LaTeX row for GPU timing column in `drafts/accuracy_results_for_paper.md`

### 4.4 Outputs

**`ANALYSIS.md`** written to the result directory:
```
# clax benchmark analysis — <DATE>
## Platform summary
## Results scorecard  (PASS ✓ / FAIL ✗ / WARN ⚠ / PENDING ⏳ per metric)
## Anomalies and recommended actions
## Comparison vs baselines
## Paper-ready numbers
## Updated LaTeX snippet
## What to watch when planck_cl lands
```

**Push notification** via `PushNotification` tool:
- Title: `clax bench <DATE>: N pass, M fail, K pending`
- Body: top-line summary (3 sentences max) + path to `ANALYSIS.md`

### 4.5 Re-entrant behavior

If `planck_cl.txt` is absent when Stage 2 first runs, it produces an `ANALYSIS.md` clearly marked `[PARTIAL — planck_cl pending]`. When the watcher detects `planck_cl.txt` has arrived, it re-runs both stages and produces a final `ANALYSIS.md` marked `[COMPLETE]`. The notification fires again with the complete picture.

---

## 5. Watcher — `scripts/watch_benchmark.py`

### 5.1 Logic

```
1. Compute today's result dir: /ptmp/minh/clax-bench/<DATE>/
2. If dir doesn't exist → exit 0 (jobs not started)
3. Glob *.txt files; compute set of scripts with results
4. Read .last_analyzed mtime if present
5. If any .txt file is newer than .last_analyzed → proceed
6. Run: python scripts/analyze_benchmark.py --dir <dir>
7. Run: python scripts/agent_analyze_benchmark.py --dir <dir>
8. Touch .last_analyzed
9. If all 7 expected files present and all analyzed → CronDelete self
```

### 5.2 Expected files list

```python
EXPECTED = ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]
```

### 5.3 Self-cancellation

`watch_benchmark.py` is given its own cron ID at creation time (passed via `--cron-id` flag or read from `~/.clax_bench_cron_id`). When all 7 files are present and Stage 2 has successfully written `ANALYSIS.md` marked `[COMPLETE]`, it calls the CronDelete API to remove itself.

### 5.4 Error handling

- If Stage 1 fails (parse error, missing dep), logs to `/ptmp/minh/clax-bench/<DATE>/watcher.log` and sends a notification: `"clax bench watcher error — check watcher.log"`
- If Stage 2 fails (API error, timeout), still keeps `REPORT.md` from Stage 1 and retries Stage 2 on the next watcher tick
- Never crashes silently — all exceptions caught, logged, notified

---

## 6. File layout in repo

```
scripts/
  analyze_benchmark.py           Stage 1
  agent_analyze_benchmark.py     Stage 2
  watch_benchmark.py             Watcher + self-cancel logic
docs/superpowers/specs/
  2026-05-04-benchmark-analysis-pipeline-design.md   this file
```

No new dependencies beyond `anthropic` (already available in `py311forge`) and the standard library (`re`, `json`, `pathlib`, `subprocess`, `argparse`).

---

## 7. Constraints and non-goals

- **No GPU required** — all three scripts run on the login node (raven01–04)
- **No JAX** — Stage 1 is pure Python stdlib; Stage 2 is Claude API only
- **No new conda env** — `py311forge` already has `anthropic` (to verify at install time)
- **Not a dashboard** — output is Markdown files, not a web UI
- **Not a regression tracker** — no historical database; each run is self-contained
- **planck_cl graceful degradation** — the pipeline is useful before `planck_cl.txt` lands

---

## 8. Open questions (resolved)

| Question | Decision |
|---|---|
| Trigger mechanism | Combined: scheduled watcher (30 min) + manual invocation |
| Polling interval | 30 minutes |
| Agent reasoning depth | Full narrative + anomaly detection + paper numbers |
| Self-cancel when done | Yes — CronDelete after all 7 files analyzed |
| Re-run on planck_cl arrival | Yes — idempotent, re-runs both stages |
| Push notification | Yes — PushNotification after Stage 2 each time |
