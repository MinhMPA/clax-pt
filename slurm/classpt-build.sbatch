#!/bin/bash -l
#SBATCH --job-name=classpt-build
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# Partition: main, as Part 0's classpt CPU template says (and
# slurm/bench-v100-igpu.sbatch:78). `main` is HIDDEN from plain `sinfo` and
# `scontrol show partition` -- only `sinfo -a` lists it (`main up infinite 8`)
# -- so do not "correct" it to gpu. It is the one that schedules:
# `sbatch --test-only --partition=main` starts immediately on igpu01, while
# --partition=gpu estimates a ~30 h wait. No --gres: build and smokes are CPU.
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ptval}
cd "$REPO"

echo "=== A2 Step 2: build (scripts/setup_classpt_env.sh) ==="
bash scripts/setup_classpt_env.sh

echo "=== provenance ==="
python -c "import classy, numpy, scipy, sys; print('classy', classy.__file__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| py', sys.version.split()[0])"

echo "=== A2 Step 3: legacy-fiducial oracle probe ==="
PYTHONPATH="$REPO" python - <<'EOF'
import numpy as np
from classy import Class
from scripts import validation_cosmologies as vc
# Guard probe: get_Pd2d2_0 before compute()/initialize_output must raise, not
# return the zero-initialised cdef double -- a silent 0.0 would let clax's
# missing +0.25 b2^2 Pd2d2_0 term agree with the oracle for the wrong reason.
try: Class().get_Pd2d2_0()
except Exception as e: print("expected error:", type(e).__name__)
else: raise SystemExit("ERROR get_Pd2d2_0 is unguarded: returned instead of raising")
prm = vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=False, yhe=None)
M = Class(); M.set(prm); M.compute()
h = M.h(); k_h = vc.ept_kgrid_numpy(); k = k_h * h
M.initialize_output(k, 0.38, len(k_h))
print("ap", M.get_ap_ratios(0.38), "f", M.scale_independent_growth_factor_f(0.38), "Pd2d2_0", M.get_Pd2d2_0())
try: M.get_ap_ratios(0.5)
except Exception as e: print("expected error:", type(e).__name__)
EOF

echo "=== A2 PASS ==="
