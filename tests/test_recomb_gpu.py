"""S1 GPU-only (smsharma/clax#48): thermodynamics tangents agree between CPU and
GPU to < 1e-10, normwise (max|Δ| / max|cpu| per table and parameter). Must run in
its OWN pytest session on a GPU node: several other test modules pin JAX to CPU at
import, which pins the whole session. The session needs BOTH backends, so
JAX_PLATFORMS must include cpu next to cuda (e.g. ``JAX_PLATFORMS=cuda,cpu``);
with ``JAX_PLATFORMS=cuda`` alone the CPU backend is never initialised. The test
SKIPS, with the missing backend named in the reason, when either backend is
unavailable. Multi-cosmology: recomb_downstream_cosmology (3 points)."""
import jax
import numpy as np
import pytest

from tests.recomb_test_utils import QS, tangent_tables, thermo_prec

pytestmark = pytest.mark.slow


def _missing_backend():
    """Name of the first JAX backend this test needs that is unavailable, else None."""
    for name in ("gpu", "cpu"):
        try:
            if not jax.devices(name):
                return name
        except RuntimeError:
            return name
    return None


_MISSING = _missing_backend()


@pytest.mark.skipif(
    _MISSING is not None,
    reason=f"needs both GPU and CPU JAX backends (missing: {_MISSING}); "
           "run on a GPU node with JAX_PLATFORMS=cuda,cpu",
)
@pytest.mark.parametrize("theta", ["h", "omega_b", "Y_He"])
def test_cpu_gpu_tangents_agree(recomb_downstream_cosmology, theta):
    name, p = recomb_downstream_cosmology
    prec = thermo_prec(20000)
    with jax.default_device(jax.devices("cpu")[0]):
        _, t_cpu = tangent_tables(p, prec, theta)
    with jax.default_device(jax.devices("gpu")[0]):
        _, t_gpu = tangent_tables(p, prec, theta)
    worst = 0.0
    for q in QS:
        a, b = np.asarray(t_cpu[q]), np.asarray(t_gpu[q])
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), \
            f"ERROR non-finite {q} tangent ({name}, {theta})"
        # Normwise, the metric of the design study's CPU-vs-GPU comparison (not in the repo): a pointwise
        # ratio on entries 1e-15..1e-23 of the table maximum measures round-off, not agreement.
        worst = max(worst, float(np.max(np.abs(b - a)) / max(np.max(np.abs(a)), 1e-300)))
    print(f"{name} d/d{theta}: CPU-GPU tangent normwise (max|Δ| / max|cpu|) {worst:.3e}")
    assert worst < 1e-10, f"ERROR CPU vs GPU tangents normwise {worst:.3e} ({name}, {theta})"
