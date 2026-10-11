#!/bin/bash
#SBATCH --job-name=clax-rsa
#SBATCH --output=/lustre/work/n2minh/std/clax/rsa/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/rsa/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen.111@gmail.com

# Validation of the RSA-trigger fix (branch fix/rsa-trigger), modelled on
# slurm/preset-validation-v100.sbatch. Submit from the login node, e.g.
#   WT=/lustre/work/n2minh/clax-wt-rsa SECTIONS=1234 BANDS=/path/bands_60_120.py \
#       sbatch --export=ALL --job-name=clax-rsa-after slurm/rsa-trigger-v100.sbatch
# Sections: 1 new tests (sources vs CLASS, high-l preset, thermodynamics) in full mode;
#           2 full tests/test_preset_accuracy.py + tests/test_bessel_table.py;
#           3 fast suite with the known pre-existing failure deselected;
#           4 timings (scripts/benchmark_speed.py planck_cl) and the 60/120 k-per-decade bands.
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
WT=${WT:-/lustre/work/n2minh/clax-wt-rsa}
SECTIONS=${SECTIONS:-1234}
FILTER='^WARNING\|^I0000\|^W0000\|^E0000'
cd "${WT}" || exit 1
echo "Node: $(hostname)  Worktree: ${WT} @ $(git rev-parse --short HEAD 2>/dev/null)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$SECTIONS" == *1* ]]; then
echo "=== 1. new tests (full mode) ==="
pytest tests/test_thermodynamics.py -k tau_free_streaming tests/test_sources_vs_class.py \
    "tests/test_preset_accuracy.py::TestPlanckClPreset::test_fiducial_massless_recfast_high_l" \
    -q -o addopts="" -p no:cacheprovider -rA -s 2>&1 | grep -v "${FILTER}"
fi
if [[ "$SECTIONS" == *2* ]]; then
echo "=== 2. preset accuracy + Bessel table (full mode) ==="
pytest tests/test_bessel_table.py tests/test_preset_accuracy.py -q -o addopts="" -p no:cacheprovider -rA -s 2>&1 | grep -v "${FILTER}"
fi
if [[ "$SECTIONS" == *3* ]]; then
echo "=== 3. fast suite ==="
pytest tests/ --fast -q -o addopts="" -p no:cacheprovider \
    --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5 2>&1 | grep -v "${FILTER}" | tail -15
fi
if [[ "$SECTIONS" == *4* ]]; then
echo "=== 4. timings and 60/120 k-per-decade bands ==="
python scripts/benchmark_speed.py planck_cl 2>&1 | grep -v "${FILTER}" | tail -25
if [[ -n "${BANDS:-}" ]]; then python "${BANDS}" 60 120 2>&1 | grep -v "${FILTER}"; fi
fi
echo "End: $(date)"
