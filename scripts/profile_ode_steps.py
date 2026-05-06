#!/usr/bin/env python3
"""Profile actual ODE step counts for each PrecisionParams preset.

Run on an igpu V100 node via slurm/bench-v100-profile-steps.sbatch.
Output lines of the form:
    [STEPS] caller=pt_mpk num_steps=[4 7 5 3 ...]
are printed as side effects of jax.debug.print inside perturbations.py.

Usage (from /lustre/work/n2minh/clax):
    python scripts/profile_ode_steps.py 2>&1 | tee profile_steps_out.txt
"""
import os
import sys
sys.path.insert(0, ".")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk

PRESETS = {
    "fit_cl":    PrecisionParams.fit_cl(),
    "fast_cl":   PrecisionParams.fast_cl(),
    "medium_cl": PrecisionParams.medium_cl(),
    "planck_cl": PrecisionParams.planck_cl(),
}

PARAM_SETS = {
    "fiducial":       CosmoParams(),
    "high_omega_b":   CosmoParams(omega_b=0.0264),
    "low_omega_b":    CosmoParams(omega_b=0.0183),
    "high_omega_cdm": CosmoParams(omega_cdm=0.144),
    "low_omega_cdm":  CosmoParams(omega_cdm=0.096),
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
