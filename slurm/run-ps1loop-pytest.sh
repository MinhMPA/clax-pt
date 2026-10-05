#!/bin/bash
# Usage (login node): slurm/run-ps1loop-pytest.sh <pytest args...>
# Submits slurm/ps1loop-tests.sbatch for THIS worktree, waits, prints the tail and ERROR lines,
# and exits with the job's status.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/lustre/work/n2minh/std/clax/ps1loop
mkdir -p "$OUT"
jid=$(sbatch --parsable --wait --export=ALL,CLAX_REPO="$PWD",PYTEST_ARGS="$*" slurm/ps1loop-tests.sbatch)
rc=$?
jid=${jid%%;*}
echo "job $jid exit $rc"
tail -n 25 "$OUT/ps1loop-tests.out.$jid"
grep -h "ERROR" "$OUT/ps1loop-tests.out.$jid" | head -n 20
exit $rc
