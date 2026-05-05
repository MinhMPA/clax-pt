#!/bin/bash
#SBATCH --job-name=clax-bench-planck
#SBATCH --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen.111@gmail.com

# --- Environment ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax

# Expose bundled NVIDIA CUDA libraries to the dynamic linker.
# jax[cuda12] pip wheels install libcusparse/libcublas/etc. under
# site-packages/nvidia/*/lib/, which is not on LD_LIBRARY_PATH by default.
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Sanity ---
echo "=== Platform ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX', jax.__version__)"

# --- Output directory ---
cd /lustre/work/n2minh/clax
DATE=$(date +%F)
OUT=/lustre/work/n2minh/clax/benchmark/${DATE}/$(hostname)-v100
mkdir -p "$(dirname "${OUT}")"

echo ""
echo "=== Starting planck_cl benchmark: ${OUT} ==="
echo "Start: $(date)"

python scripts/benchmark_speed.py planck_cl > "${OUT}-planck_cl.txt" 2>&1

echo "End: $(date)"

# --- Append planck_cl result to REPORT.md if it exists ---
REPORT=/lustre/work/n2minh/clax/benchmark/${DATE}/REPORT.md
{
  echo ""
  echo "## planck_cl (SLURM job ${SLURM_JOB_ID})"
  grep -h "median\|cached\|PASS\|FAIL" "${OUT}-planck_cl.txt" 2>/dev/null | head -20
  echo ""
  echo "### $(basename "${OUT}-planck_cl.txt") (last 5 lines)"
  tail -5 "${OUT}-planck_cl.txt" 2>/dev/null || echo "(not found)"
} >> "${REPORT}"

echo "planck_cl appended to: ${REPORT}"
