#!/bin/bash
#SBATCH --job-name=clax-profile-steps-v2
#SBATCH --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen.111@gmail.com

eval "$(micromamba shell hook --shell bash)"
micromamba activate clax

NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=== Platform ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX', jax.__version__)"

cd /lustre/work/n2minh/clax
echo "=== ODE step profiling v2 start: $(date) ==="
python scripts/profile_ode_steps_v2.py 2>&1
echo "=== Done: $(date) ==="
