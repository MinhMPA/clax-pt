#!/bin/bash -l
#SBATCH --job-name=ptval-e2e-ap
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
#
# End-to-end AP gate: clax background -> thermodynamics -> perturbations ->
# compute_ept_from_clax(omfid=0.31, field="cb") against CLASS-PT's nine spectra.
# Complements the stage sweep (tests/test_ept_multicosmo.py), which feeds
# compute_ept a reference pk_lin instead of solving for it.
#
# PTVAL_E2E_CASES=a,b    -> just those cosmologies (the sharding knob: one case
#                           costs a full perturbation solve, ~45 min measured on
#                           job 14668, and the pipeline cache is per-case, so
#                           splitting by case across nodes loses no work).
# PTVAL_E2E_SUBSET=fast  -> FAST_CASES x FAST_Z (3 cases x 1 z).
# neither                -> the full 14 cases x 3 z sweep (~11 h serially).
# -s so the per-case [pipeline] timings stream instead of being swallowed by
# pytest capture on pass; --durations for the per-test breakdown at the end.
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
# LD_LIBRARY_PATH loop copied verbatim from slurm/bench-v100-igpu.sbatch.
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax}
cd "$REPO"
export PYTHONPATH="$REPO"
echo "Node: $(hostname)  Commit: $(git rev-parse --short HEAD)  Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "cases: ${PTVAL_E2E_CASES:-<all>}   subset: ${PTVAL_E2E_SUBSET:-<none>}   prec: ${PTVAL_E2E_PREC:-fast}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices', jax.devices())"
python -c "import clax.ept as e; assert len(e._GAUSS_NODES) == 40, 'gauss_tab.dat missing: 10-pt fallback active'; print('GL nodes', len(e._GAUSS_NODES))"

LOG=/lustre/work/n2minh/std/clax/ptval/${SLURM_JOB_NAME}.pytest.${SLURM_JOB_ID}.txt
set +e
python -m pytest tests/test_ept_e2e_multicosmo.py -m slow \
    --durations=30 --durations-min=1.0 --verbosity=2 -rs -s \
    -p no:cacheprovider > "${LOG}" 2>&1
rc=$?
set -e
tail -n 45 "${LOG}"
echo "full transcript: ${LOG}"
test "${rc}" -eq 0 && echo PASS || { echo "FAIL rc=${rc}"; exit "${rc}"; }
