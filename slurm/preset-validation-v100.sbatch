#!/bin/bash
#SBATCH --job-name=clax-preset-val
#SBATCH --output=/lustre/work/n2minh/std/clax/presets/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/presets/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen.111@gmail.com

# Validation of the Oct 2026 preset retune (branch fix/preset-retune):
#   1. new accuracy tests (slow, full mode) + Bessel-table test
#   2. per-stage timings of planck_cl and fit_cl (scripts/benchmark_speed.py)
#   3. the fast suite with the known pre-existing failure deselected
# Submit from the login node: sbatch slurm/preset-validation-v100.sbatch
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
WT=${WT:-/lustre/work/n2minh/clax-wt-presets}
SECTIONS=${SECTIONS:-123}   # subset of 1 (tests) 2 (timings) 3 (fast suite)
cd "${WT}" || exit 1
echo "Node: $(hostname)  Worktree: ${WT} @ $(git rev-parse --short HEAD 2>/dev/null)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$SECTIONS" == *1* ]]; then
echo "=== 1. accuracy tests (full mode) ==="
pytest tests/test_bessel_table.py tests/test_preset_accuracy.py -q -o addopts="" -p no:cacheprovider -rA -s 2>&1 | grep -v "^WARNING\|^I0000\|^W0000"
fi
if [[ "$SECTIONS" == *2* ]]; then
echo "=== 2. timings ==="
python scripts/benchmark_speed.py planck_cl 2>&1 | grep -v "^WARNING\|^I0000\|^W0000" | tail -25
python scripts/benchmark_speed.py fit_cl 2>&1 | grep -v "^WARNING\|^I0000\|^W0000" | tail -25
fi
if [[ "$SECTIONS" == *3* ]]; then
echo "=== 3. fast suite ==="
pytest tests/ --fast -q -o addopts="" -p no:cacheprovider \
    --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5 2>&1 | grep -v "^WARNING\|^I0000\|^W0000" | tail -15
fi
echo "End: $(date)"
