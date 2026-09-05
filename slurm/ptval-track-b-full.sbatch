#!/bin/bash -l
#SBATCH --job-name=ptval-track-b-full
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
#
# Track B full-mode gate (clax-pt validation, Part 1b). Complement of
# ptval-fast-suite.sbatch: no --fast, so the multi-cosmology fixtures sweep all
# grid points and every @pytest.mark.slow test runs. The login node may only run
# the bounded --fast subset (Part 0 "Compute placement"), so this is the gate.
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
# LD_LIBRARY_PATH loop copied verbatim from slurm/bench-v100-igpu.sbatch (differs
# from the Part 0 doc's CONDA_PREFIX-glob version — Part 0 requires this file's
# copy to match bench-v100-igpu.sbatch, not the doc's template text).
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ptval}
cd "$REPO"
export PYTHONPATH="$REPO"
echo "Node: $(hostname)  Commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices', jax.devices())"
python -c "import clax.ept as e; assert len(e._GAUSS_NODES) == 40, 'gauss_tab.dat missing: 10-pt fallback active'; print('GL nodes', len(e._GAUSS_NODES))"

# Per-test names into a transcript so the per-grid-point results survive the
# tail; stdout keeps the last 40 lines plus the PASS/FAIL marker the gate greps
# for. --verbosity=2 (not -v): pyproject.toml addopts is "-q", and -q/-v are
# relative, so they would cancel to the default dot output.
LOG=/lustre/work/n2minh/std/clax/ptval/${SLURM_JOB_NAME}.pytest.${SLURM_JOB_ID}.txt
set +e
python -m pytest tests/test_ap.py tests/test_ept_ap.py tests/test_ept.py \
    tests/test_ept_assembly.py tests/test_ept_accuracy.py \
    tests/test_ept_gradients.py tests/test_ept_h_channels.py \
    tests/test_ir_resummation_classpt.py tests/test_ir_resummation_jax.py \
    --verbosity=2 -rs -p no:cacheprovider > "${LOG}" 2>&1
rc=$?
set -e
tail -n 40 "${LOG}"
echo "AP gradient grid points passed:  LCDM(h,omega_cdm)=$(grep -c 'test_gradients_match_finite_differences\[.*PASSED' "${LOG}" || true)  w0waCDM(+w0,wa)=$(grep -c 'test_w0wa_gradients_match_finite_differences\[.*PASSED' "${LOG}" || true)"
echo "full transcript: ${LOG}"
test "${rc}" -eq 0 && echo PASS || { echo "FAIL rc=${rc}"; exit "${rc}"; }
