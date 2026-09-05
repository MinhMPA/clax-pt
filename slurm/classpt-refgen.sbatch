#!/bin/bash -l
#SBATCH --job-name=classpt-refgen
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# Part 0 classpt CPU template (docs/superpowers/plans/2026-09-03-clax-pt-validation-part0-common.md).
# `main` is HIDDEN from plain `sinfo`/`scontrol show partition` (only `sinfo -a`
# lists it) but schedules immediately -- do not "correct" it to gpu (Ruling 5,
# also slurm/classpt-build.sbatch). No --gres: this job is CPU-only (env classpt,
# no clax/jax import). No git operations here beyond the read-only rev-parse echo.
#
# Task A5: full CLASS-PT reference generation for the clax-pt validation
# campaign (spec §4.8) -- 14 distinct cases x Z_LIST=(0,0.38,0.8) at the
# campaign default (ap=yes, cb=yes, omfid=0.31, bias=fiducial), plus four
# diagnostic files at z=0.38. No --class-extra anywhere here (that flag is
# reserved for the A4 provenance gate, not the campaign/diagnostic runs).
set -uo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ptval}
cd "$REPO"
export PYTHONPATH="$REPO"
echo "Node: $(hostname)  Commit: $(git rev-parse --short HEAD)"
python -c "import classy, numpy; print('classy', classy.__file__, 'numpy', numpy.__version__)"

GEN="python scripts/generate_classpt_reference.py"
FAILED=()
# Precision-boundary bug found & confirmed in job-14290/14291/14292/14293
# diagnostics (see task-A5-report.md): CLASS-PT's nonlinear_pt_init hits
# array_interpolate_spline(..., log(tau_req[i_z]), ...) with x slightly >
# x_max (nonlinear_pt.c:1324/1829) whenever z=0 is combined with other z's
# in the SAME z_pk list, for 6/14 cosmologies (background-dependent, fully
# deterministic -- reproduced identically at OMP_NUM_THREADS=1 and 8).
# Ruling 19 (coordinator): every campaign case is generated per single z --
# three separate `--z-list <z>` calls, uniformly, for ALL cases (not just
# the 6 that hit the bug) -- so every z0.*_ap_omfid0.31_cb.npz file in the
# campaign has the SAME provenance (CLASS samples sources from z_max_pk and
# spline-interpolates P_L in ln tau per request; a combined 3-z call and a
# single-z call build slightly different interpolation grids). No fallback
# logic, no physics input changes: same generator, same default CLASS
# params, no --class-extra.
for c in $($GEN --list-distinct); do
  for z in 0 0.38 0.8; do
    $GEN --cosmology "$c" --z-list "$z" || FAILED+=("$c:z$z")
  done
done
$GEN --cosmology lcdm_fiducial --z-list 0.38 --cb no        || FAILED+=("lcdm_fiducial:cb-no")
$GEN --cosmology lcdm_fiducial --z-list 0.38 --ap no        || FAILED+=("lcdm_fiducial:ap-no")
$GEN --cosmology lcdm_fiducial --z-list 0.38 --bias nonzero || FAILED+=("lcdm_fiducial:biasnz")
SKIPPED=()
$GEN --cosmology w0wa_m07_m10 --z-list 0.38 --use-ppf no --tag noppf \
  || SKIPPED+=("w0wa_m07_m10/z0.380_ap_omfid0.31_cb_noppf.npz: use_ppf=no rejected or failed (see job log)")
python scripts/write_classpt_manifest.py --skipped "${SKIPPED[@]}"
echo "FAILED: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
