#!/bin/bash -l
#SBATCH --job-name=ps1loop-ladder
#SBATCH --output=/lustre/work/n2minh/std/clax/ps1loop/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ps1loop/%x.err.%j
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
# Convergence ladder for ps_1loop's settings (spec §7; Task 9 of the clax-pt vs ps_1loop_jax validation):
#   sbatch slurm/ps1loop-ladder.sbatch [--points case:z ...] [--rungs name ...]
# Same environment as slurm/ps1loop-tests.sbatch (env clax, CPU by default -- set JAX_PLATFORMS=cuda to
# override); appends to test_logs/ps1loop/errors.jsonl (layer "ladder") in the worktree and prints one line
# per (point, rung), plus one REMAINDER line per rung at (lcdm_fiducial, z = 0.38).
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ps1loop}
cd "$REPO"
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
echo "HEAD: $(git -C "$REPO" rev-parse --short HEAD)  ps_1loop: $(git -C /home/n2minh/ps_1loop_jax-for-pfs rev-parse --short HEAD)"
git -C "$REPO" status --porcelain | head -20
python -c "import jax, ps_1loop_jax; print('devices', jax.devices())"
echo "ladder args: $*"
python scripts/ps1loop_convergence_ladder.py "$@"
echo DONE
