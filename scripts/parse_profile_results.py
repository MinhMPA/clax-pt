#!/usr/bin/env python3
"""Parse profiling outputs and print recommendation tables.

Usage:
    python scripts/parse_profile_results.py \
        --steps-v2   <path to profile_ode_steps_v2 output> \
        --compile    <path to profile_compile_time output> \
        --steps-v1   /lustre/work/n2minh/std/clax/benchmark/clax-profile-steps.out.5713

Prints:
  1. Merged step-count table (v1 + v2) with recommended ode_max_steps
  2. Compile-time table (c1 = compile+run, c2 = run only)
  3. Pre-plan action table for params.py
"""
import argparse
import math
import re
from collections import defaultdict

SAFETY = {
    "fit_cl":      1.5,
    "fast_cl":     1.5,
    "medium_cl":   1.5,
    "planck_fast": 1.5,
    "science_cl":  2.0,
    "planck_cl":   2.0,
}
CURRENT_CEILING = {
    "fit_cl":      1024,
    "fast_cl":     65536,
    "medium_cl":   65536,
    "planck_fast": 65536,
    "science_cl":  131072,
    "planck_cl":   131072,
}


def next_pow2(n):
    return 2 ** math.ceil(math.log2(max(n, 1)))


def parse_steps(path):
    results = defaultdict(lambda: defaultdict(int))
    cur_preset = cur_param = None
    with open(path) as f:
        for line in f:
            m = re.match(r'\[PROFILE(?:_TENSOR)?\] preset=(\S+)\s+params=(\S+)', line)
            if m:
                cur_preset, cur_param = m.group(1), m.group(2)
            m2 = re.match(r'\[STEPS\] caller=\S+ num_steps=(\d+)', line)
            if m2 and cur_preset and cur_param:
                v = int(m2.group(1))
                if v > results[cur_preset][cur_param]:
                    results[cur_preset][cur_param] = v
    return results


def parse_compile(path):
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for line in f:
            m = re.match(r'\[TIME\]\t(\S+)\t(\S+)\t(\S+)', line)
            if m:
                tag, preset, val = m.group(1), m.group(2), float(m.group(3))
                suffix = "_c1" if tag.endswith("_c1") else "_c2"
                base = tag[:-3]
                data[base][preset][suffix] = val
    return data


def merge_steps(v1_path, v2_path):
    merged = defaultdict(lambda: defaultdict(int))
    for path in [v1_path, v2_path]:
        if not path:
            continue
        for preset, pdata in parse_steps(path).items():
            for param, v in pdata.items():
                if v > merged[preset][param]:
                    merged[preset][param] = v
    return merged


def print_step_table(merged):
    print("\n" + "="*80)
    print("=== ODE Step Counts — Combined Results (v1+v2) ===")
    print(f"{'Preset':<15} {'Max obs':>10} {'Ceiling':>10} {'Util%':>7} {'Safety':>7} {'Recommended':>12}  Action")
    print("-"*80)
    for preset in ["fit_cl","fast_cl","medium_cl","planck_fast","science_cl","planck_cl"]:
        pdata = merged.get(preset, {})
        if not pdata:
            print(f"{preset:<15} {'(no data)':>10}")
            continue
        obs_max = max(pdata.values())
        ceiling = CURRENT_CEILING[preset]
        util = obs_max / ceiling
        safety = SAFETY[preset]
        recommended = next_pow2(obs_max * safety)
        if obs_max >= ceiling * 0.98:
            action = "FLAG: solver hit limit — INCREASE ceiling"
        elif recommended >= ceiling:
            action = "already optimal"
        else:
            action = f"REDUCE {ceiling}→{recommended} (×{ceiling//recommended})"
        print(f"{preset:<15} {obs_max:>10,} {ceiling:>10,} {util:>7.1%} {safety:>7.1f} {recommended:>12,}  {action}")


def print_compile_table(data):
    print("\n" + "="*80)
    print("=== Compile+Run (c1) and Run-only (c2) Times in seconds ===")
    print("Compile estimate = c1 - c2\n")
    tags = ["bg","th","pt","hr","grad_pk","jacrev_pk","ept","grad_ept"]
    presets = ["fit_cl","fast_cl","medium_cl","planck_fast","science_cl","planck_cl"]
    for tag in tags:
        if tag not in data:
            continue
        print(f"  [{tag}]")
        print(f"  {'Preset':<15} {'c1 (s)':>10} {'c2 (s)':>10} {'compile est (s)':>16}")
        for preset in presets:
            d = data[tag].get(preset, {})
            c1 = d.get("_c1", float("nan"))
            c2 = d.get("_c2", float("nan"))
            ce = c1 - c2 if (c1==c1 and c2==c2) else float("nan")
            print(f"  {preset:<15} {c1:>10.1f} {c2:>10.1f} {ce:>16.1f}")
        print()


def print_preplan(merged, data):
    print("\n" + "="*80)
    print("=== PRE-PLAN: Actions for params.py ===")
    print("""
STEP-COUNT ACTIONS (Task 6A of this plan):
  For each preset listed as REDUCE above, update clax/params.py:
    PrecisionParams.<preset>(): set ode_max_steps = recommended value.

  After update:
    pytest tests/ --fast -x -q
  If TestMassiveNu.test_pk_at_k005 fails → go to Branch B (add failing
  params to profile sweep, compute new ceiling).

COMPILE-TIME INTERPRETATION:
  1. If grad_pk compile estimate (c1-c2) for fast_cl > 5 min:
       Reducing ode_max_steps 65536→2048 removes 5 checkpoint tree levels
       and should cut this by ~3-5x. Apply step-count reduction first.

  2. If grad_pk compile estimate for planck_cl > 20 min:
       131072→32768 (4x reduction, 2 fewer tree levels). Apply step-count
       reduction first, then re-run this profile script to verify.

  3. If ept compile estimate (c1-c2) > 5 min:
       EPT is algebra-only (no ODE). Slow compile means FFTLog kernels or
       _load_matrices() are being re-traced inside JIT.
       Fix: ensure _load_matrices(prec.nmax) is called outside jax.jit
       (e.g., cache with functools.lru_cache or close over static arrays).

  4. If grad_ept c1 >> ept c1 + grad_pk c1 for the same preset:
       The EPT gradient graph is expanding unexpectedly. Check whether
       _ir_resummation_numpy() is inside the trace (it uses numpy, so it
       will cause retracing). Ensure _ir_precomputed= is always supplied
       when calling compute_ept inside jax.grad.

  5. If jacrev_pk c1 > 20 min for fit_cl:
       This is the Fisher Jacobian baseline at 50 k-modes × 5 params.
       Cold-start cost may be unavoidable without jax.compilation_cache.
       Check whether enabling the JAX persistent compilation cache
       reduces the second-run cost to near-zero.

TENSOR MODES:
  After measuring tensor step counts, check if tensor_perturbations_solve
  also reads ode_max_steps. If so, tensor modes may have a separate
  optimal ceiling that can be set independently.
  Current code: tensor_perturbations_solve() uses prec.ode_max_steps (same
  field as scalar). If tensor max_steps << scalar max_steps, consider adding
  a separate prec.ode_max_steps_tensor field to params.py.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-v2",  default=None)
    ap.add_argument("--steps-v1",  default=None)
    ap.add_argument("--compile",   default=None)
    args = ap.parse_args()

    merged = merge_steps(args.steps_v1, args.steps_v2)
    if merged:
        print_step_table(merged)

    compile_data = {}
    if args.compile:
        compile_data = parse_compile(args.compile)
        print_compile_table(compile_data)

    print_preplan(merged, compile_data)
