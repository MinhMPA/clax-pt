#!/bin/bash -l
#SBATCH -J clax-bench-planck
#SBATCH -o /ptmp/minh/std/clax/benchmark/%x.out.%j
#SBATCH -e /ptmp/minh/std/clax/benchmark/%x.err.%j
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp

# --- Environment ---
# Modules: anaconda/3/2023.03 (latest on Raven), cuda/12.2 (CUDA 12.x)
# Env: py311forge (clax_class-pt_py310forge does not exist on this cluster;
#       py311forge has clax+clax.ept+jax-cuda12 installed)
module purge
module load anaconda/3/2023.03 cuda/12.2
conda activate /u/minh/conda-envs/py311forge

# Expose bundled NVIDIA CUDA libraries to the dynamic linker.
# Required because jax[cuda12] pip wheels install libcusparse/libcublas/etc.
# under site-packages/nvidia/*/lib/, which is not on LD_LIBRARY_PATH by default.
NVIDIA_SP=/u/minh/conda-envs/py311forge/lib/python3.11/site-packages/nvidia
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
cd /u/minh/clax
DATE=$(date +%F)
OUT=/ptmp/minh/clax-bench/${DATE}/$(hostname)-a100
mkdir -p "$(dirname "${OUT}")"

echo ""
echo "=== Starting planck_cl benchmark: ${OUT} ==="
echo "Start: $(date)"

python scripts/benchmark_speed.py planck_cl > "${OUT}-planck_cl.txt" 2>&1

echo "End: $(date)"

# --- Append planck_cl result to REPORT.md if it exists ---
REPORT=/ptmp/minh/clax-bench/${DATE}/REPORT.md
{
  echo ""
  echo "## planck_cl (SLURM job ${SLURM_JOB_ID})"
  grep -h "median\|cached\|PASS\|FAIL" "${OUT}-planck_cl.txt" 2>/dev/null | head -20
  echo ""
  echo "### $(basename "${OUT}-planck_cl.txt") (last 5 lines)"
  tail -5 "${OUT}-planck_cl.txt" 2>/dev/null || echo "(not found)"
} >> "${REPORT}"

echo "planck_cl appended to: ${REPORT}"
