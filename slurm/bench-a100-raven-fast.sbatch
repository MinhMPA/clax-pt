#!/bin/bash -l
#SBATCH -J clax-bench-fast
#SBATCH -o /ptmp/minh/std/clax/benchmark/%x.out.%j
#SBATCH -e /ptmp/minh/std/clax/benchmark/%x.err.%j
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=02:00:00
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
python -c "import clax; import clax.ept; print('clax+ept OK')"

# --- Output directory ---
cd /u/minh/clax
DATE=$(date +%F)
OUT=/ptmp/minh/clax-bench/${DATE}/$(hostname)-a100
mkdir -p "$(dirname "${OUT}")"

echo ""
echo "=== Starting fast benchmark suite: ${OUT} ==="
echo "Start: $(date)"

python scripts/benchmark_solvers.py   --n-warmup 2 --n-repeat 5  > "${OUT}-solvers.txt"    2>&1
python scripts/benchmark_gradients.py --n-warmup 2 --n-repeat 5  > "${OUT}-gradients.txt"  2>&1
python scripts/benchmark_speed.py     fit_cl                     > "${OUT}-fit_cl.txt"     2>&1
python scripts/benchmark_ept.py       --preset medium            > "${OUT}-ept.txt"        2>&1
python scripts/benchmark_clpp.py      --preset medium --l-max 2500 > "${OUT}-clpp.txt"    2>&1
python scripts/accuracy_classpt.py                               > "${OUT}-accuracy.txt"   2>&1

echo ""
echo "=== Fast suite complete: $(date) ==="

# --- Partial report (fast scripts only) ---
REPORT=/ptmp/minh/clax-bench/${DATE}/REPORT.md
{
  echo "# clax benchmark report — ${DATE}"
  echo ""
  echo "## Platform"
  echo "- Node: $(hostname)"
  echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
  echo "- JAX: $(python -c 'import jax; print(jax.__version__)')"
  echo "- Branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "- Commit: $(git rev-parse --short HEAD)"
  echo "- SLURM Job IDs: fast=${SLURM_JOB_ID} planck_cl=see bench-a100-raven-planck_cl.sbatch"
  echo ""
  echo "## Headline numbers (fast suite)"
  grep -h "median\|Backward/forward\|max rel err\|cached\|PASS\|FAIL" "${OUT}"-*.txt 2>/dev/null | head -60
  echo ""
  echo "## Per-script tail (last 5 lines each)"
  for f in "${OUT}"-solvers.txt "${OUT}"-gradients.txt "${OUT}"-fit_cl.txt \
            "${OUT}"-ept.txt "${OUT}"-clpp.txt "${OUT}"-accuracy.txt; do
    echo ""
    echo "### $(basename "$f")"
    tail -5 "$f" 2>/dev/null || echo "(not found)"
  done
  echo ""
  echo "## planck_cl"
  echo "(pending — see ${OUT}-planck_cl.txt when job completes)"
} > "${REPORT}"

echo "Partial report written to: ${REPORT}"
