"""Test fixtures for clax test suite.

Provides:
- Session-scoped pipeline fixtures (run expensive solves ONCE)
- Reference data loading from CLASS-generated .npz files
- Default CosmoParams and PrecisionParams
- --fast flag for quick regression checks

Pipeline fixtures (session-scoped, shared across test files):
    pipeline_fast_cl      — CosmoParams() + fast_cl preset
    pipeline_fast_cl_k5   — CosmoParams() + fast_cl + pt_k_max_cl=5.0
"""

import json
import os

# Enable 64-bit JAX (required for recombination numerics)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from dataclasses import replace as _dc_replace

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve

# Path to reference data
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


# ---------------------------------------------------------------------------
# Session-scoped pipeline fixtures
# ---------------------------------------------------------------------------
# Each perturbation solve takes 2-5 min on CPU.  By sharing results across
# all test files that use the same precision settings, the full test suite
# runs in ~10 min instead of 60+.

@pytest.fixture(scope="session")
def pipeline_fast_cl():
    """Background + thermo + perturbations with fast_cl preset.

    Used by: test_harmonic, test_high_l, test_lensing, test_cl_pp_implementations.
    Returns (params, prec, bg, th, pt).
    """
    params = CosmoParams()
    prec = _dc_replace(PrecisionParams.fast_cl(), pt_k_chunk_size=20)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


@pytest.fixture(scope="session")
def pipeline_fast_cl_k5():
    """Background + thermo + perturbations with fast_cl + k_max=5.

    Extends the k-grid for source-based Limber and Halofit sigma(R).
    Used by: test_cl_pp_source_limber, test_clpp_limber_accuracy,
             test_lensing_nonlinear, test_clpp_halofit_ratio.
    Returns (params, prec, bg, th, pt).
    """
    params = CosmoParams()
    prec = _dc_replace(PrecisionParams.fast_cl(),
                       pt_k_max_cl=5.0, pt_k_chunk_size=20)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False,
        help="Run fast subset of tests (every 10th point)"
    )


@pytest.fixture
def fast_mode(request):
    return request.config.getoption("--fast")


@pytest.fixture
def lcdm_bg_ref():
    """Load LCDM fiducial background reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'background.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_scalars():
    """Load LCDM fiducial scalar quantities."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'scalars.json')
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def lcdm_derived():
    """Load LCDM fiducial derived parameters."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'derived.json')
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def lcdm_thermo_ref():
    """Load LCDM fiducial thermodynamics reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'thermodynamics.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_cls_ref():
    """Load LCDM fiducial C_l reference data (unlensed)."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'cls.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_cls_lensed_ref():
    """Load LCDM fiducial lensed C_l reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'cls_lensed.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_pk_ref():
    """Load LCDM fiducial P(k) reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'pk.npz')
    return dict(np.load(path, allow_pickle=True))


def relative_error(computed, reference, eps=1e-30):
    """Compute relative error, avoiding division by zero."""
    return np.abs(computed - reference) / (np.abs(reference) + eps)


def max_relative_error(computed, reference, eps=1e-30):
    """Return (max_rel_err, index_of_max)."""
    rel = relative_error(computed, reference, eps)
    idx = np.argmax(rel)
    return float(rel[idx]), int(idx)


def assert_close(computed, reference, rtol, name="quantity", coordinate=None):
    """Assert computed matches reference within rtol, with clear error message.

    Follows CLAUDE.md principle: concise, actionable error messages.
    """
    max_err, idx = max_relative_error(computed, reference)
    if max_err > rtol:
        coord_str = f" at index {idx}"
        if coordinate is not None:
            coord_str = f" at {coordinate[idx]:.6g}"
        msg = (
            f"{name}: max rel error {max_err:.4%}{coord_str}"
            f" (expected {reference[idx]:.6e}, got {computed[idx]:.6e})"
            f" -- tolerance {rtol:.4%}"
        )
        raise AssertionError(msg)
