#!/bin/bash -l
#SBATCH --job-name=ptval-p1-deltacb
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices', jax.devices())"
python -c "import clax.ept as e; assert len(e._GAUSS_NODES) == 40, 'gauss_tab.dat missing: 10-pt fallback active'; print('GL nodes', len(e._GAUSS_NODES))"
# Full mode (no --fast): runs the `slow` perturbation solves that --fast skips,
# and the full nulcdm_cosmology sweep. `-s` keeps the measured suppression and
# ncdm-off identity numbers printed by TestDeltaCb visible in the job log.
# NOT `-q -s`: that pair suppresses pytest's "N passed" summary line (verified
# locally on the login node), leaving only progress dots. `-s --tb=short` keeps
# the prints AND the summary.
python -m pytest tests/test_perturbations.py -s --tb=short -k TestDeltaCb -p no:cacheprovider 2>&1 | tail -n 40
test "${PIPESTATUS[0]}" -eq 0 && echo PASS
