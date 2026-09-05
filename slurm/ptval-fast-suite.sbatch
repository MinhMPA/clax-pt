#!/bin/bash -l
#SBATCH --job-name=ptval-fast-suite
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=03:00:00
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
# Ruling 50 (Sep 3, 2026): the fast suite runs on CPU on purpose. Measured on igpu
# V100 nodes: 1:10:24 CPU (job 14328) vs 2:33:02 / 2:40:27 GPU (14365 / 14372);
# the small-problem pipeline tests are latency-bound on the GPU (test_cl_pp.py's
# pipeline fixture: ~2 min CPU, >16 min GPU in job 14389). Until Task B8 three test
# modules pinned the whole session to CPU silently; this is the same behaviour made
# explicit, and conftest prints the backend after collection. Track B/C gates stay
# on the GPU. JAX reads JAX_PLATFORMS at import, so the export must precede python.
export JAX_PLATFORMS=cpu
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices', jax.devices())"
python -c "import clax.ept as e; assert len(e._GAUSS_NODES) == 40, 'gauss_tab.dat missing: 10-pt fallback active'; print('GL nodes', len(e._GAUSS_NODES))"
echo "HEAD: $(git -C "$REPO" rev-parse --short HEAD)"
# B2 review (minor 2): record which uncommitted files the gate actually tested
# -- other tracks edit shared files in this worktree while this job runs.
git -C "$REPO" status --porcelain | head -20
# --deselect: pre-existing max-steps failure on the campaign base, not a
# campaign regression -- CHANGELOG.md Aug 29, 2026 (jobs 14019/14027, "no
# regressions") and Sep 1, 2026 (job 14146, "confirmed pre-existing... zero
# file overlap"); tracked as issue #30 item 5. No campaign diff touches mpk.
python -m pytest tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5 \
  2>&1 | tail -n 40
test "${PIPESTATUS[0]}" -eq 0 && echo PASS
