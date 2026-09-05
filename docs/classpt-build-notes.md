# CLASS-PT build notes (Task A2)

CLASS-PT `09d5531a` built into the dedicated micromamba env `classpt`
(Python 3.10.21, NumPy 1.26.4, Cython 0.29.37, SciPy 1.15.2, conda-forge
OpenBLAS 0.3.34, conda-forge gcc/g++ 16.2.0) by `scripts/setup_classpt_env.sh`.
Run it through `slurm/classpt-build.sbatch` — measured wall time for the whole
job (clean rebuild + both smokes) is **57 s** on one igpu node with
`--cpus-per-task=8`.

Three retries were needed. Each cause and its fix:

1. **OpenBLAS path is hard-coded in three places, only one of them
   env-readable.** Part 0 finding 17 names `Makefile:50` and
   `python/setup.py:24` (`OPENBLAS_PATH`), but `make classy` (`Makefile:216-223`)
   runs `pip install .` at the repo root, so the file that actually builds the
   extension is the **root** `setup.py`, whose `openblas_dir`
   (`setup.py:57`, reused in `extra_link_args` at `setup.py:64`) is a literal
   `/share/software/user/open/openblas/0.3.28/lib` with no environment hook.
   Fix, entirely through the environment (no tracked CLASS-PT file edited):
   `OPENBLAS=-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lopenblas` as a
   make variable for the `class` link, plus `LIBRARY_PATH` and `LDFLAGS`
   exports so the linker finds and rpaths `$CONDA_PREFIX/lib/libopenblas` for
   the Cython extension. `OPENBLAS_PATH` is exported too, for the case where
   `python/setup.py` is used instead. Verified: `ldd _classy…so` resolves
   `libopenblas.so.0` to the env, and `import classy` works with
   `LD_LIBRARY_PATH` unset.

2. **`pip`'s build isolation silently overrode the env pins.**
   `Makefile:217` is a bare `pip install .`, and `pyproject.toml` requires
   unpinned `numpy`/`cython`, so pip built against NumPy 2 / Cython 3 in
   `/tmp/pip-build-env-*/overlay`. `PIP_NO_BUILD_ISOLATION=1` does **not** fix
   this. `--no-build-isolation` is `action="store_false"` on
   `dest=build_isolation`, so the environment variable carries the *dest*
   value. Measured on pip 26.2.1 with a throwaway package, counting
   `Installing build dependencies: started`:

   | setting | isolation used |
   |---|---|
   | (none) | yes |
   | `PIP_NO_BUILD_ISOLATION=1` / `=yes` | yes |
   | `PIP_NO_BUILD_ISOLATION=false` / `=0` | **no** |
   | explicit `--no-build-isolation` | no |

   Fix: `export PIP_NO_BUILD_ISOLATION=false`. The wheel is now built against
   this env's NumPy 1.26.4 / Cython 0.29.37.

3. **`abs()` on a C double does not compile in `classy.pyx`.**
   `python/classy.pyx:18` is `from libc.stdlib cimport *`, which binds `abs`
   to C `int abs(int)`. The `get_ap_ratios` body specified by Task A2's brief
   used `abs(self.nlpt.z_pk[i] - z) < 1e-10` and failed with

   ```
   Error compiling Cython file:
   ...
           for i in range(self.nlpt.z_pk_num):
               if abs(self.nlpt.z_pk[i] - z) < 1e-10:
                                        ^
   python/classy.pyx:4614:37: Cannot assign type 'double' to 'int'
   ```

   Fix in `scripts/classpt_patches/classy_ap_ratios.patch`: the same predicate
   spelled as a chained C comparison, `-1e-10 < self.nlpt.z_pk[i] - z < 1e-10`,
   which needs no extra `cimport` and leaves the tolerance, the return value
   and the `CosmoSevereError` behaviour unchanged.

The patch also adds `self._check_pt()` as the first statement of
`get_Pd2d2_0` — a deliberate departure from the brief's verbatim body.
`_check_pt` (`classy.pyx:4768` post-patch) raises on both `nlpt.method == 0`
and `not self.output_init`, and every other PT accessor calls it (e.g.
`pk_mm_real` at `:4816`). Unguarded, `get_Pd2d2_0()` before
`initialize_output()` returns the zero-initialised `cdef double`, i.e. `0.0`,
which would let clax's missing `+0.25 b2^2 Pd2d2_0` term agree with the oracle
for the wrong reason. `slurm/classpt-build.sbatch` exercises the guard on a
fresh `Class()` before `compute()`.

The gcc >= 10 `-fno-common` "multiple definition" failure the brief anticipated
did **not** occur: conda-forge gcc 16.2.0 compiles CLASS-PT `09d5531a` clean at
`OPTFLAG = -O4 -ffast-math`, so no `-fcommon` is needed.

`__CLASSDIR__` (`Makefile:71-72`) is baked as `/home/n2minh/CLASS-PT`, which is
where `nonlinear_pt.c:861-983` finds `pt_matrices/`. The root `setup.py`'s
`classy_builder.build_extension` re-invokes `make libclass.a` with
`CLASSDIR="."`; that is harmless only because the script builds `libclass.a`
first, so the inner make is a no-op ("`libclass.a` is up to date"). The import
smoke therefore runs from the repo root, not from `$CLASSPT_DIR`, so a relative
`__CLASSDIR__` would be caught.

**Use `--partition=main`, and do not "correct" it.** `main` is a *hidden*
partition here: plain `sinfo` and `scontrol show partition` list only `gpu`,
but `sinfo -a` shows `main up infinite 8` over the same `igpu[01-08]`. It is
also the partition that actually schedules —

```
$ sbatch --test-only --partition=main slurm/classpt-build.sbatch
sbatch: Job 14281 to start at 2026-09-03T07:59:40 using 8 processors on nodes igpu01 in partition main
$ sbatch --test-only --partition=gpu  slurm/classpt-build.sbatch
sbatch: Job 14282 to start at 2026-09-04T13:59:40 using 8 processors on nodes igpu01 in partition gpu
```

— a ~30 h difference. `slurm/bench-v100-igpu.sbatch:78` already documents
"Partition: main", and the campaign's other jobs ran there.
`slurm/classpt-build.sbatch` uses `--partition=main` with no `--gres`, since
the build and its smokes are pure CPU.

Verification at the legacy fiducial (`slurm/classpt-build.sbatch`, job 14272):

```
classy OK: /home/n2minh/micromamba/envs/classpt/lib/python3.10/site-packages/classy/__init__.py
classy ... | numpy 1.26.4 | scipy 1.15.2 | py 3.10.21
ap (1.0020530866394164, 0.9990316747989566, 0.7166475183766424) f 0.716647518376645 Pd2d2_0 3837.2596970939717
expected error: CosmoSevereError
```
