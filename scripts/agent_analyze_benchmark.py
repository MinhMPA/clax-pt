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

PREFLIGHT_BASELINE = {
    "kvaerno5_median_s": 13.09,
    "platform": "gpu",
    "node": "ravg1002",
    "job_id": "26878115",
}

BENCHMARK_TARGETS = """
## BENCHMARK.md §4 Thresholds (GPU A100 targets)

### §4.1 Timing
- fit_cl total cached: ≤ 50 s  (current baseline: 34 s on V100, CLAUDE.md)
- planck_cl total cached: ≤ 600 s  (baseline: 487 s on H100)
- Perturbation solve: ≤ 30 s on V100/A100 (fit_cl preset)
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

- P_mm real: max 0.31%, mean 0.04%  ← CPU reference
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
One paragraph: node, GPU model, JAX version, branch/commit.

## Results scorecard
A table with every metric: metric name | value | status (PASS ✓ / FAIL ✗ / WARN ⚠ / PENDING ⏳) | threshold | notes.
For PENDING metrics (planck_cl if missing), say what we're waiting for.

## Anomalies and recommended actions
For each FAIL or WARN: what the number is, what the threshold is, probable cause, recommended action.
If everything passes, say so explicitly and note any metrics unusually close to thresholds.

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
A100 (40 GB) & clax \\texttt{{fit\\_cl}} & JAX 0.10.0 & {{fit_cl_total:.0f}} s & {{planck_cl_total}} s & \\\\
```
If planck_cl is pending, use "pending".

## What to watch when planck_cl lands
2–3 bullet points: which metrics will become available, what values to expect based on CLAUDE.md, what would constitute a regression.

{BENCHMARK_TARGETS}

{ACCURACY_PAPER_CONTEXT}

## PREFLIGHT BASELINE (job 26878115, gpudev A100, 2026-05-04)
- kvaerno5 median (1 warmup, 1 repeat): 13.09 s  ← note: includes partial JIT overhead; production runs use 2 warmup + 5 repeat
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, driver 580.126.20
- JAX: 0.10.0, [CudaDevice(id=0)]

Be precise and cite numbers. Do not hedge. If a metric is borderline, say so. If the pipeline is healthy, say so clearly. The user will read this before they open their laptop.
Mark the document [COMPLETE] if planck_cl data is present, [PARTIAL — planck_cl pending] if not.
"""


def load_context(result_dir: Path) -> tuple[str, str]:
    """Load results.json and all available raw .txt files."""
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
    print("\nFirst 20 lines:")
    print("\n".join(analysis.split("\n")[:20]))


if __name__ == "__main__":
    main()
