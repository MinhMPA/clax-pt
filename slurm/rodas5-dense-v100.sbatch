#!/bin/bash
#SBATCH --job-name=clax-r5dense
#SBATCH --output=/lustre/work/n2minh/std/clax/rodas5dense/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/rodas5dense/%x.err.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen.111@gmail.com

# Rodas5 dense-output fix: GPU validation and timing (Oct 12, 2026).
# Submit from the login node with variables in the environment, e.g.
#   MODE=cltest PYTEST_ARGS="-k lcdm_fiducial" sbatch --export=ALL --job-name=r5d-cl slurm/rodas5-dense-v100.sbatch
#   MODE=timing SOLVER=rodas5 TAG=before sbatch --export=ALL --job-name=r5d-t slurm/rodas5-dense-v100.sbatch
#   MODE=fast sbatch --export=ALL --job-name=r5d-fast slurm/rodas5-dense-v100.sbatch
# Env: MODE (cltest | timing | abc | preset | fast), VARIANTS/REPEAT (abc), CONFIGS (preset), SOLVER (timing), TAG (timing results tag),
#      DIAG_ARGS (extra args for the timing script), PYTEST_ARGS (cltest),
#      COMMIT (run from a `git archive` export of this commit instead of the
#      live worktree, so later edits cannot leak into a queued job),
#      WT (worktree; default /lustre/work/n2minh/clax-wt-rodas5dense).

eval "$(micromamba shell hook --shell bash)"
micromamba activate clax

NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

: "${MODE:?set MODE=cltest|timing|abc|preset|fast}"
WT=${WT:-/lustre/work/n2minh/clax-wt-rodas5dense}
OUTDIR=/lustre/work/n2minh/std/clax/rodas5dense
mkdir -p "${OUTDIR}"

if [ -n "${COMMIT:-}" ]; then
    SRC="${OUTDIR}/src-${COMMIT}-${SLURM_JOB_ID}"
    mkdir -p "${SRC}"
    git -C "${WT}" archive "${COMMIT}" | tar -x -C "${SRC}"
    cd "${SRC}" || exit 1
    GIT_DESC="${COMMIT} (archive export)"
else
    cd "${WT}" || exit 1
    GIT_DESC="$(git rev-parse --short HEAD) (live worktree)"
fi

echo "=== Platform ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Source: $(pwd) @ ${GIT_DESC}"
echo "Mode: ${MODE}   Start: $(date)"

case "${MODE}" in
  cltest)
    pytest tests/test_rodas5_dense_output.py -q -o addopts="" -p no:cacheprovider -s ${PYTEST_ARGS:-}
    status=$?
    ;;
  timing)
    : "${SOLVER:?set SOLVER=rodas5|kvaerno5}"
    OUT="${OUTDIR}/timing-${TAG:-run}-${SOLVER}-${SLURM_JOB_ID}.json"
    python diags/diag_rodas5_dense_output.py --solver "${SOLVER}" --out "${OUT}" ${DIAG_ARGS:-}
    status=$?
    echo "results: ${OUT}"
    ;;
  abc)
    : "${VARIANTS:?set VARIANTS='name=path ...'}"
    OUT="${OUTDIR}/abc-${TAG:-run}-${SLURM_JOB_ID}.json"
    python diags/diag_rodas5_dense_output.py --abc ${VARIANTS} --repeat "${REPEAT:-4}" --out "${OUT}"
    status=$?
    echo "results: ${OUT}"
    ;;
  preset)
    OUT="${OUTDIR}/preset-${TAG:-run}-${SLURM_JOB_ID}.json"
    python diags/diag_preset_solver_choice.py --configs "${CONFIGS:-A,B,C,D,E,F}" --out "${OUT}"
    status=$?
    python diags/diag_preset_solver_choice.py --report "${OUT}" --md "${OUT%.json}.md" || true
    echo "results: ${OUT}"
    ;;
  fast)
    pytest tests/ --fast -q -o addopts="" -p no:cacheprovider \
        --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5
    status=$?
    ;;
  *)
    echo "unknown MODE=${MODE}"; status=2 ;;
esac

echo "End: $(date) (exit ${status})"
# Tidy up only this job's own git-archive export (created above), never the worktree.
if [ -n "${COMMIT:-}" ] && [ -d "${SRC}" ]; then
    cd "${OUTDIR}" && python -c "import shutil, sys; shutil.rmtree(sys.argv[1])" "${SRC}"
fi
exit ${status}
