#!/bin/bash -l
#SBATCH --job-name=ptval-track-c
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
#
# Track C gate (clax-pt validation, Part 2, Task C0): ept_inputs_from_clax /
# compute_ept_from_clax(omfid, field) wiring tests, including the slow
# delta_cb physics tests (perturbations_solve at m_ncdm = 0.06/0.15/0.30 eV).
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

ARGS=${PTVAL_PYTEST_ARGS:-"tests/test_ept_from_clax.py -q"}
echo "pytest args: $ARGS"
python -m pytest $ARGS -p no:cacheprovider -rs 2>&1 | tail -n 40
echo "PASS"
