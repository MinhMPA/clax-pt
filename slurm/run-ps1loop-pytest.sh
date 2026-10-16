#!/bin/bash
# Usage (login node): slurm/run-ps1loop-pytest.sh <pytest args...>
# Submits slurm/ps1loop-tests.sbatch for THIS worktree, waits, prints the tail and ERROR lines
# (from the full pytest log), and exits with the job's status. Arguments reach pytest losslessly
# (one per line in a file; quoting such as -k "a and b" and commas survive).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/lustre/work/n2minh/std/clax/ps1loop
mkdir -p "$OUT"
argfile=$(mktemp "$OUT/ps1loop-args.XXXXXX")
trap 'rm -f "$argfile"' EXIT   # safe: sbatch --wait returns only after the job has read the file
printf '%s\n' "$@" > "$argfile"
jid=$(sbatch --parsable --wait --export=ALL,CLAX_REPO="$PWD",PYTEST_ARGS_FILE="$argfile" slurm/ps1loop-tests.sbatch)
rc=$?
jid=${jid%%;*}
if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch failed (exit $rc)"
    exit "$rc"
fi
echo "job $jid exit $rc"
tail -n 25 "$OUT/ps1loop-tests.out.$jid"
grep -h "ERROR" "$OUT/ps1loop-tests.pytest.$jid.log" | head -n 20
exit $rc
