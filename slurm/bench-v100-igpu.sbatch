#!/bin/bash -l
#SBATCH --job-name=clax-validation
#SBATCH --output=/lustre/work/n2minh/std/clax/benchmark/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/benchmark/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp

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
python -c "import clax; import clax.ept; print('clax+ept OK')"

# --- Output directories ---
cd /lustre/work/n2minh/clax
DATE=$(date +%F)
SCRATCH_OUT=/lustre/work/n2minh/clax/benchmark/${DATE}
OUT=${SCRATCH_OUT}/$(hostname)-v100
mkdir -p "${SCRATCH_OUT}" /lustre/work/n2minh/std/clax/benchmark

echo ""
echo "=== Starting igpu V100 validation + benchmark: ${OUT} ==="
echo "Partition time limit: infinite (--time=7-00:00:00 ceiling)"
echo "Start: $(date)"

# --- Validation (gap vs Raven A100) ---
echo "--- pytest --fast suite ---"
python -m pytest tests/ -v --fast -q                                   > "${OUT}-pytest_fast.txt"  2>&1

echo "--- Full validation notebook ---"
EXECUTED_NB=${SCRATCH_OUT}/clax-pt_full_validation_executed.ipynb
jupyter nbconvert --to notebook --execute \
    notebooks/clax-pt_full_validation.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    --output "${EXECUTED_NB}"                                           > "${OUT}-notebook.txt"    2>&1

# --- V100 timing (A100 vs V100 paper comparison) ---
echo "--- Timing benchmarks ---"
python scripts/benchmark_solvers.py   --n-warmup 2 --n-repeat 5        > "${OUT}-solvers.txt"     2>&1
python scripts/benchmark_gradients.py --n-warmup 2 --n-repeat 5        > "${OUT}-gradients.txt"   2>&1
python scripts/benchmark_speed.py     fit_cl                           > "${OUT}-fit_cl.txt"      2>&1
python scripts/benchmark_ept.py       --preset medium                  > "${OUT}-ept.txt"         2>&1
python scripts/benchmark_clpp.py      --preset medium --l-max 2500     > "${OUT}-clpp.txt"        2>&1
python scripts/accuracy_classpt.py                                     > "${OUT}-accuracy.txt"    2>&1

echo ""
echo "=== All scripts complete: $(date) ==="

# --- REPORT.md ---
REPORT=${SCRATCH_OUT}/REPORT.md
{
  echo "# clax igpu V100 benchmark + validation report — ${DATE}"
  echo ""
  echo "## Cluster / partition"
  echo "- Cluster: igpu (Kavli IPMU)"
  echo "- Partition: main (single partition; reported time limit: infinite)"
  echo "- GPU gres: gpu:1 (Tesla V100-SXM2-32GB)"
  echo "- Node: $(hostname)"
  echo ""
  echo "## Platform"
  echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
  echo "- JAX: $(python -c 'import jax; print(jax.__version__)')"
  echo "- Branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "- Commit: $(git rev-parse --short HEAD)"
  echo "- SLURM Job ID: ${SLURM_JOB_ID}"
  echo ""
  echo "## Validation (gap vs Raven A100)"
  echo ""
  echo "### pytest --fast"
  PYTEST_SUMMARY=$(tail -3 "${OUT}-pytest_fast.txt" 2>/dev/null)
  echo "\`\`\`"
  echo "${PYTEST_SUMMARY}"
  echo "\`\`\`"
  if echo "${PYTEST_SUMMARY}" | grep -q "failed\|error"; then
    echo "**STATUS: FAIL**"
  else
    echo "**STATUS: PASS**"
  fi
  echo ""
  echo "### Notebook execution"
  NB_TAIL=$(tail -5 "${OUT}-notebook.txt" 2>/dev/null)
  echo "\`\`\`"
  echo "${NB_TAIL}"
  echo "\`\`\`"
  if echo "${NB_TAIL}" | grep -qi "error\|exception\|traceback"; then
    echo "**STATUS: FAIL** — see ${OUT}-notebook.txt"
  else
    echo "**STATUS: PASS** — executed notebook at ${EXECUTED_NB}"
  fi
  echo ""
  echo "## V100 timing (A100 vs V100 paper comparison)"
  echo ""
  grep -h "median\|Backward/forward\|max rel err\|cached\|PASS\|FAIL\|Total\|fit_cl\|planck_cl" \
       "${OUT}"-solvers.txt "${OUT}"-gradients.txt "${OUT}"-fit_cl.txt \
       "${OUT}"-ept.txt "${OUT}"-clpp.txt "${OUT}"-accuracy.txt 2>/dev/null | head -80
  echo ""
  echo "## Per-script tail (last 5 lines each)"
  for f in "${OUT}"-pytest_fast.txt "${OUT}"-notebook.txt \
            "${OUT}"-solvers.txt "${OUT}"-gradients.txt "${OUT}"-fit_cl.txt \
            "${OUT}"-ept.txt "${OUT}"-clpp.txt "${OUT}"-accuracy.txt; do
    echo ""
    echo "### $(basename "$f")"
    tail -5 "$f" 2>/dev/null || echo "(not found)"
  done
} > "${REPORT}"

echo "Report written to: ${REPORT}"
