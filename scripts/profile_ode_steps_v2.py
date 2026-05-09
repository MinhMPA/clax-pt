#!/usr/bin/env python3
"""Profile ODE step counts — gap-fill run (v2).

Covers presets missing from the May-6 job 5713 run:
  - science_cl  (never profiled)
  - planck_fast (never profiled)
  - planck_cl   remaining 4/8 param sets (timed out in job 5713)
  - tensor modes for science_cl, planck_cl (fiducial + massive_nu)

Usage (from /lustre/work/n2minh/clax on benchmark/clax-pt):
    python scripts/profile_ode_steps_v2.py 2>&1 | tee v2_steps_out.txt
"""
import sys
sys.path.insert(0, ".")

import jax
jax.config.update("jax_enable_x64", True)

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk, tensor_perturbations_solve

SCALAR_PRESETS = {
    "science_cl":  PrecisionParams.science_cl(),
    "planck_fast": PrecisionParams.planck_fast(),
    "planck_cl":   PrecisionParams.planck_cl(),
}

# For tensor modes: science_cl and planck_cl.
# planck_fast has identical perturbation settings to planck_cl — skip.
TENSOR_PRESETS = {
    "science_cl": PrecisionParams.science_cl(),
    "planck_cl":  PrecisionParams.planck_cl(),
}

# planck_cl: only run the 4 param sets that timed out in job 5713
PLANCK_CL_REMAINING = ["low_omega_cdm", "high_h", "low_h", "massive_nu"]

ALL_PARAMS = {
    "fiducial":       CosmoParams(),
    "high_omega_b":   CosmoParams(omega_b=0.0264),
    "low_omega_b":    CosmoParams(omega_b=0.0183),
    "high_omega_cdm": CosmoParams(omega_cdm=0.144),
    "low_omega_cdm":  CosmoParams(omega_cdm=0.096),
    "high_h":         CosmoParams(h=0.741),
    "low_h":          CosmoParams(h=0.574),
    "massive_nu":     CosmoParams(m_ncdm=0.06),
}


def run_scalar(preset_name, prec, param_name, params):
    print(f"\n{'='*60}", flush=True)
    print(f"[PROFILE] preset={preset_name}  params={param_name}", flush=True)
    print(f"[PROFILE] ode_max_steps={prec.ode_max_steps}", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    perturbations_solve_mpk(params, prec, bg, th)
    print(f"[DONE] preset={preset_name} params={param_name}", flush=True)


def run_tensor(preset_name, prec, param_name, params):
    print(f"\n{'='*60}", flush=True)
    print(f"[PROFILE_TENSOR] preset={preset_name}  params={param_name}", flush=True)
    print(f"[PROFILE_TENSOR] ode_max_steps={prec.ode_max_steps}", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    tensor_perturbations_solve(params, prec, bg, th)
    print(f"[DONE_TENSOR] preset={preset_name} params={param_name}", flush=True)


if __name__ == "__main__":
    print("Platform:", jax.devices(), flush=True)
    print("JAX", jax.__version__, flush=True)

    # science_cl and planck_fast: all 8 param sets
    for preset_name in ["science_cl", "planck_fast"]:
        prec = SCALAR_PRESETS[preset_name]
        for param_name, params in ALL_PARAMS.items():
            try:
                run_scalar(preset_name, prec, param_name, params)
            except Exception as exc:
                print(f"[ERROR] {preset_name}/{param_name}: {exc}", flush=True)

    # planck_cl: only the 4 that timed out in job 5713
    prec_planck = SCALAR_PRESETS["planck_cl"]
    for param_name in PLANCK_CL_REMAINING:
        try:
            run_scalar("planck_cl", prec_planck, param_name, ALL_PARAMS[param_name])
        except Exception as exc:
            print(f"[ERROR] planck_cl/{param_name}: {exc}", flush=True)

    # Tensor modes: fiducial + massive_nu only
    for preset_name, prec in TENSOR_PRESETS.items():
        for param_name in ["fiducial", "massive_nu"]:
            try:
                run_tensor(preset_name, prec, param_name, ALL_PARAMS[param_name])
            except Exception as exc:
                print(f"[ERROR_TENSOR] {preset_name}/{param_name}: {exc}", flush=True)

    print("\n[PROFILE_V2 COMPLETE]", flush=True)
