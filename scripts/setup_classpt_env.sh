#!/bin/bash -l
# scripts/setup_classpt_env.sh — build the CLASS-PT oracle in a dedicated env.
# Idempotent: re-running skips env creation and already-applied patches.
# NEVER installs into any other env (carpile/cosmopower/cosmodesi/fli-mf-nuts).
#
# The build is CPU-bound (a full `make clean` rebuild every run); measured at
# 53-57 s wall with MAKE_JOBS=8. Run it through slurm/classpt-build.sbatch
# rather than on the login node.
set -euo pipefail

ENV_NAME=classpt
CLASSPT_DIR=${CLASSPT_DIR:-/home/n2minh/CLASS-PT}
CLASSPT_COMMIT=09d5531a
REPO=$(cd "$(dirname "$0")/.." && pwd)
PATCHES="$REPO/scripts/classpt_patches"
MAKE_JOBS=${MAKE_JOBS:-${SLURM_CPUS_PER_TASK:-4}}

eval "$(micromamba shell hook --shell bash)"
if [ ! -d "$HOME/micromamba/envs/$ENV_NAME" ]; then
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 "numpy<2" "cython<3" \
    scipy "setuptools<60" gcc gxx make pip wheel openblas
fi
micromamba activate "$ENV_NAME"

cd "$CLASSPT_DIR"
git rev-parse --short HEAD | grep -q "^$CLASSPT_COMMIT" || { echo "ERROR CLASS-PT HEAD is not $CLASSPT_COMMIT"; exit 1; }
for p in classy_ap_ratios classy_kh_units; do
  if git apply --reverse --check "$PATCHES/$p.patch" 2>/dev/null; then
    echo "patch $p already applied"
  else
    git apply "$PATCHES/$p.patch" && echo "applied $p"
  fi
done

# CLASS-PT hard-codes an OpenBLAS path that does not exist on this cluster, in
# three places: Makefile:50 (OPENBLAS), setup.py:57 (openblas_dir -- the one the
# `classy` target actually uses, via `pip install .` at Makefile:217) and
# python/setup.py:24 (OPENBLAS_PATH, the only env-readable one). No tracked file
# is edited; every override below goes through the environment or a make var.
#   OPENBLAS=...           overrides Makefile:50 for the `class` link
#   LIBRARY_PATH/LDFLAGS   let the linker find $CONDA_PREFIX/lib/libopenblas
#                          despite the dead -L baked into setup.py:57/64, and
#                          bake an rpath so classy imports without
#                          LD_LIBRARY_PATH
#   OPENBLAS_PATH          honoured by python/setup.py if it is ever used
#   PIP_NO_BUILD_ISOLATION makes `pip install .` build against this env's
#                          numpy<2 / cython<3 instead of pulling unpinned
#                          numpy 2 / Cython 3 into an isolated build env.
#                          The value MUST be false/0: pip's --no-build-isolation
#                          is action="store_false" on dest=build_isolation, so
#                          the env var carries the *dest* value -- measured with
#                          pip 26.2.1, `=false` matches the CLI flag (no
#                          "Installing build dependencies" line) while `=1`
#                          silently leaves isolation ON.
BLAS_LINK="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lopenblas"
export OPENBLAS_PATH="$CONDA_PREFIX/lib" CC=gcc CXX=g++
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib ${LDFLAGS:-}"
export PIP_NO_BUILD_ISOLATION=false
export PIP_DISABLE_PIP_VERSION_CHECK=1

make clean >/dev/null 2>&1 || true
make -j"$MAKE_JOBS" OPENBLAS="$BLAS_LINK" CC=gcc PYTHON=python class libclass.a
make OPENBLAS="$BLAS_LINK" CC=gcc PYTHON=python classy

# Import smoke, run from $REPO: nonlinear_pt.c:861-983 opens pt_matrices via the
# compile-time __CLASSDIR__ (Makefile:71-72), so a wrong/relative CLASSDIR would
# only show up outside $CLASSPT_DIR.
cd "$REPO"
python - <<'EOF'
from classy import Class
M = Class()
assert hasattr(M, "get_ap_ratios") and hasattr(M, "get_Pd2d2_0"), "patched accessors missing"
print("classy OK:", __import__("classy").__file__)
EOF
