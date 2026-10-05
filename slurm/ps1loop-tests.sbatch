#!/bin/bash -l
#SBATCH --job-name=ps1loop-tests
#SBATCH --output=/lustre/work/n2minh/std/clax/ps1loop/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ps1loop/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# clax-pt vs ps_1loop_jax validation (docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md).
# One pytest invocation; arguments come from $PYTEST_ARGS. No git network operations here
# (compute nodes have no SSH key). CPU by default, as in slurm/ptval-fast-suite.sbatch
# (Ruling 50: small problems are latency-bound on the V100); set JAX_PLATFORMS=cuda to override.
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
OUT=/lustre/work/n2minh/std/clax/ps1loop
mkdir -p "$OUT"
# Arguments: PYTEST_ARGS_FILE (one argument per line, lossless; written by slurm/run-ps1loop-pytest.sh)
# or, for direct sbatch use, PYTEST_ARGS (whitespace-split, no quoting).
if [ -n "${PYTEST_ARGS_FILE:-}" ]; then
    mapfile -t ARGS < "$PYTEST_ARGS_FILE"
else
    read -r -a ARGS <<< "${PYTEST_ARGS:?set PYTEST_ARGS_FILE or PYTEST_ARGS}"
fi
LOG="$OUT/ps1loop-tests.pytest.${SLURM_JOB_ID:-local}.log"
echo "pytest args: ${ARGS[*]}  (full log: $LOG)"
set +e
python -m pytest "${ARGS[@]}" > "$LOG" 2>&1
rc=$?
set -e
tail -n 60 "$LOG"
if [ "$rc" -eq 0 ]; then echo PASS; fi
exit "$rc"
