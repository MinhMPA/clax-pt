#!/usr/bin/env python3
"""Profile compile time vs run time for the clax pipeline (per preset).

Two-call timing methodology:
    jax.jit(f)(x1)  -> first call: trace + compile + run    (c1)
    jax.jit(f)(x2)  -> second call: run only (warm cache)   (c2)
    compile_estimate ~ c1 - c2

We change the *traced* CosmoParams (not the static PrecisionParams) between
calls so the cached executable is reused on the second call.

Output format (TSV, parseable):
    [TIME]\\t<tag>\\t<preset>\\t<seconds>

Tags emitted (per preset that succeeds):
    forward stages : bg_c1/bg_c2, th_c1/th_c2, pt_c1/pt_c2, hr_c1/hr_c2
    grad   P(k)    : grad_pk_c1, grad_pk_c2     (HMC presets)
    jacrev P(k)    : jacrev_pk_c1, jacrev_pk_c2 (Fisher presets)
    EPT forward    : ept_c1, ept_c2             (EPT presets)
    EPT gradient   : grad_ept_c1, grad_ept_c2   (EPT presets)

Usage (from /lustre/work/n2minh/clax):
    # SLURM/GPU:
    python scripts/profile_compile_time.py 2>&1 | tee compile_time_out.txt
    # Login-node smoke test (will not finish — kill after ~90s):
    JAX_PLATFORMS=cpu timeout 90 python scripts/profile_compile_time.py
"""
import sys
sys.path.insert(0, ".")

import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk
from clax.transfer import compute_pk_from_perturbations
from clax.harmonic import compute_cls_all, compute_cls_all_fast


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_PRESETS = [
    ("fit_cl",      PrecisionParams.fit_cl()),
    ("fast_cl",     PrecisionParams.fast_cl()),
    ("medium_cl",   PrecisionParams.medium_cl()),
    ("planck_fast", PrecisionParams.planck_fast()),
    ("science_cl",  PrecisionParams.science_cl()),
    ("planck_cl",   PrecisionParams.planck_cl()),
]

HMC_PRESETS = {"fit_cl", "fast_cl", "medium_cl", "planck_fast"}
FISHER_PRESETS = {"fit_cl", "fast_cl", "medium_cl"}
EPT_PRESETS = {"fit_cl", "fast_cl"}

FIDUCIAL = CosmoParams()
FIDUCIAL2 = CosmoParams(h=0.70, omega_cdm=0.13)  # second call — different params

PK_K_GRID = jnp.array(np.geomspace(0.01, 0.3, 50))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(tag, preset, seconds):
    """Print a single parseable timing line."""
    print(f"[TIME]\t{tag}\t{preset}\t{seconds:.6f}", flush=True)


def _time_call(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), block until ready, return (result, elapsed_s)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def _cls(pt, params, bg, prec):
    """Dispatch C_l implementation according to preset config."""
    if prec.hr_n_k_fine > 0:
        return compute_cls_all_fast(
            pt, params, bg,
            l_max=prec.hr_l_max, n_k_fine=prec.hr_n_k_fine,
        )
    return compute_cls_all(pt, params, bg, l_max=prec.hr_l_max)


def _pk_table(params, prec):
    """Full forward pipeline: params -> P(k) on PK_K_GRID at z=0."""
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve_mpk(params, prec, bg, th)
    return compute_pk_from_perturbations(pt, bg, PK_K_GRID, z=0.0)


def _ept_full(params, prec_clax):
    """Full forward pipeline through EPT components."""
    from clax.ept import compute_ept_from_clax
    bg = background_solve(params, prec_clax)
    th = thermodynamics_solve(params, prec_clax, bg)
    pt = perturbations_solve_mpk(params, prec_clax, bg, th)
    return compute_ept_from_clax(params, bg, pt, z=0.0)


# ---------------------------------------------------------------------------
# Profilers
# ---------------------------------------------------------------------------

def profile_forward(name, prec):
    """Time bg, th, pt, hr stages (compile vs warm). Stage-isolated JIT.

    Each JIT function receives the *output of the prior stage* as an explicit
    argument rather than recomputing it internally.  This ensures each timed
    call traces and compiles only its own stage, so timings are marginal (not
    cumulative).

    Call sequence:
        bg_jit(p)            -> bg result
        th_jit(p, bg)        -> th result   (bg already live, not retraced)
        pt_jit(p, bg, th)    -> pt result
        hr_jit(pt, p, bg)    -> C_l result
    """
    print(f"\n{'='*60}\n[FORWARD] {name}", flush=True)

    bg_jit = jax.jit(lambda p: background_solve(p, prec))
    th_jit = jax.jit(lambda p, bg: thermodynamics_solve(p, prec, bg))
    pt_jit = jax.jit(lambda p, bg, th: perturbations_solve_mpk(p, prec, bg, th))

    # --- bg ---
    bg1, t = _time_call(bg_jit, FIDUCIAL)
    _emit("bg_c1", name, t)
    bg2, t = _time_call(bg_jit, FIDUCIAL2)
    _emit("bg_c2", name, t)

    # --- th — takes (p, bg) so only thermodynamics_solve is traced ---
    th1, t = _time_call(th_jit, FIDUCIAL, bg1)
    _emit("th_c1", name, t)
    th2, t = _time_call(th_jit, FIDUCIAL2, bg2)
    _emit("th_c2", name, t)

    # --- pt ---
    pt1, t = _time_call(pt_jit, FIDUCIAL, bg1, th1)
    _emit("pt_c1", name, t)
    pt2, t = _time_call(pt_jit, FIDUCIAL2, bg2, th2)
    _emit("pt_c2", name, t)

    # --- hr (harmonic / C_l) — separate JIT; dispatch depends on prec ---
    try:
        hr_jit = jax.jit(lambda pt, p, bg: _cls(pt, p, bg, prec))
        cls1, t = _time_call(hr_jit, pt1, FIDUCIAL, bg1)
        _emit("hr_c1", name, t)
        cls2, t = _time_call(hr_jit, pt2, FIDUCIAL2, bg2)
        _emit("hr_c2", name, t)
    except Exception as e:
        print(f"[HR_SKIP] {name}: {e}", flush=True)


def profile_grad_pk(name, prec):
    """Time jax.grad of P(k) sum w.r.t. CosmoParams."""
    print(f"\n[GRAD_PK] {name}", flush=True)

    def _scalar(p):
        return jnp.sum(_pk_table(p, prec))

    grad_fn = jax.jit(jax.grad(_scalar))

    g1, t = _time_call(grad_fn, FIDUCIAL)
    # sanity: gradient PyTree should have an .h attribute (CosmoParams field)
    assert hasattr(g1, "h"), f"grad result missing .h attribute for {name}"
    _emit("grad_pk_c1", name, t)
    g2, t = _time_call(grad_fn, FIDUCIAL2)
    _emit("grad_pk_c2", name, t)


def profile_jacrev_pk(name, prec):
    """Time jax.jacrev of P(k) table w.r.t. CosmoParams (Fisher baseline)."""
    print(f"\n[JACREV_PK] {name}", flush=True)

    def _vec(p):
        return _pk_table(p, prec)

    jac_fn = jax.jit(jax.jacrev(_vec))

    j1, t = _time_call(jac_fn, FIDUCIAL)
    _emit("jacrev_pk_c1", name, t)
    j2, t = _time_call(jac_fn, FIDUCIAL2)
    _emit("jacrev_pk_c2", name, t)


def profile_ept(name, prec):
    """Time EPT forward and EPT gradient."""
    print(f"\n[EPT] {name}", flush=True)

    fwd = jax.jit(lambda p: _ept_full(p, prec))

    e1, t = _time_call(fwd, FIDUCIAL)
    _emit("ept_c1", name, t)
    e2, t = _time_call(fwd, FIDUCIAL2)
    _emit("ept_c2", name, t)

    def _scalar_ept(p):
        result = _ept_full(p, prec)
        return jnp.sum(result.Pk_loop)

    grad_fn = jax.jit(jax.grad(_scalar_ept))

    g1, t = _time_call(grad_fn, FIDUCIAL)
    _emit("grad_ept_c1", name, t)
    g2, t = _time_call(grad_fn, FIDUCIAL2)
    _emit("grad_ept_c2", name, t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Platform:", jax.devices(), flush=True)
    print("JAX", jax.__version__, flush=True)

    # 1. Forward pass — all presets
    for name, prec in ALL_PRESETS:
        try:
            profile_forward(name, prec)
        except Exception as e:
            print(f"[FWD_ERROR] {name}: {e}", flush=True)

    # 2. grad(P(k)) — HMC presets
    for name, prec in ALL_PRESETS:
        if name not in HMC_PRESETS:
            continue
        try:
            profile_grad_pk(name, prec)
        except Exception as e:
            print(f"[GRAD_ERROR] {name}: {e}", flush=True)

    # 3. jacrev(P(k)) — Fisher presets
    for name, prec in ALL_PRESETS:
        if name not in FISHER_PRESETS:
            continue
        try:
            profile_jacrev_pk(name, prec)
        except Exception as e:
            print(f"[JACREV_ERROR] {name}: {e}", flush=True)

    # 4. EPT forward + grad — EPT presets
    for name, prec in ALL_PRESETS:
        if name not in EPT_PRESETS:
            continue
        try:
            profile_ept(name, prec)
        except Exception as e:
            print(f"[EPT_ERROR] {name}: {e}", flush=True)

    print("\n[COMPILE_TIME_PROFILE COMPLETE]", flush=True)
