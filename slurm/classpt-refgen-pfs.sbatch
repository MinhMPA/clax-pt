#!/bin/bash -l
#SBATCH --job-name=classpt-refgen-pfs
#SBATCH --output=/lustre/work/n2minh/std/clax/ps1loop/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ps1loop/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# ps_1loop validation references (spec §3): vc.PFS_CASES x vc.PFS_Z_LIST at the campaign
# default (ap=yes, cb=yes, omfid=0.31, bias=fiducial), one z per call (Ruling 19), plus
# vc.IRNO_CASES at z=0.38 with IR resummation off and AP off. CPU-only (env classpt).
# `main` is hidden from plain sinfo but schedules (Ruling 5). No git network operations.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ps1loop}
cd "$REPO"
export PYTHONPATH="$REPO"
echo "Node: $(hostname)  Commit: $(git rev-parse --short HEAD)"
python -c "import classy, numpy; print('classy', classy.__file__, 'numpy', numpy.__version__)"
GEN="python scripts/generate_classpt_reference.py"
VC="from scripts import validation_cosmologies as vc"
FAILED=()
for c in $(python -c "$VC; print(' '.join(vc.PFS_CASES))"); do
  for z in $(python -c "$VC; print(' '.join(str(z) for z in vc.PFS_Z_LIST))"); do
    $GEN --cosmology "$c" --z-list "$z" || FAILED+=("$c:z$z")
  done
done
for c in $(python -c "$VC; print(' '.join(vc.IRNO_CASES))"); do
  $GEN --cosmology "$c" --z-list 0.38 --ap no --tag irno \
       --class-extra '{"IR resummation": "No"}' || FAILED+=("$c:irno")
done
# Keep the skipped w0wa no-PPF reference that slurm/classpt-refgen.sbatch records in MANIFEST.md, so that a
# regeneration from this script does not drop its "## Skipped" note.
python scripts/write_classpt_manifest.py \
  --skipped "w0wa_m07_m10/z0.380_ap_omfid0.31_cb_noppf.npz: use_ppf=no rejected or failed (see job log)" \
  || FAILED+=("manifest")
echo "FAILED: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
