"""Differentiable cubic spline interpolation for clax.

Provides a natural cubic spline class registered as a JAX pytree, so it can
live inside other pytrees (e.g., BackgroundResult) and flow through jit/grad/vmap.

The implementation follows DISCO-EB's approach: Thomas algorithm for the
tridiagonal system via jax.lax.fori_loop, jnp.searchsorted for interval lookup.

Key properties:
- Differentiable w.r.t. knot values y (for AD through the pipeline).
- jnp.searchsorted is not differentiated (integer output), but the polynomial
  evaluation IS differentiable w.r.t. both x_eval and the spline coefficients.

References:
    DISCO-EB: src/discoeb/spline_interpolation.py
    CLASS: tools/arrays.c (array_spline_table_*)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.tree_util.register_pytree_node_class
class CubicSpline:
    """Natural cubic spline interpolation, registered as a JAX pytree.

    Constructed from knot positions x and values y. Supports evaluation,
    first/second derivatives, and definite integrals.

    The spline satisfies: S''(x[0]) = S''(x[-1]) = 0 (natural boundary conditions).

    Attributes:
        x: knot positions, shape (N,), strictly increasing
        y: knot values, shape (N,)
        d2y: second derivatives at knots, shape (N,), from tridiagonal solve
    """

    def __init__(self, x: Float[Array, "N"], y: Float[Array, "N"]):
        """Build cubic spline from knot positions and values.

        Args:
            x: knot positions, shape (N,), must be strictly increasing
            y: knot values, shape (N,)
        """
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        self.d2y = _compute_natural_spline_coeffs(self.x, self.y)

    def evaluate(self, x_eval: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the spline at given points.

        Uses the standard cubic spline formula:
            S(x) = A*y_i + B*y_{i+1} + (A^3 - A)*d2y_i*h^2/6 + (B^3 - B)*d2y_{i+1}*h^2/6
        where A = (x_{i+1} - x) / h, B = (x - x_i) / h, h = x_{i+1} - x_i.

        Args:
            x_eval: evaluation points, any shape

        Returns:
            Spline values at x_eval, same shape as input
        """
        x_eval = jnp.asarray(x_eval)
        # Clamp to valid range
        x_clamped = jnp.clip(x_eval, self.x[0], self.x[-1])
        # Find interval indices
        idx = jnp.searchsorted(self.x, x_clamped, side="right") - 1
        idx = jnp.clip(idx, 0, len(self.x) - 2)

        # Interval quantities
        h = self.x[idx + 1] - self.x[idx]
        A = (self.x[idx + 1] - x_clamped) / h
        B = (x_clamped - self.x[idx]) / h

        # Cubic spline formula (cf. CLASS arrays.h:12, array_spline_eval macro)
        result = (
            A * self.y[idx]
            + B * self.y[idx + 1]
            + ((A**3 - A) * self.d2y[idx] + (B**3 - B) * self.d2y[idx + 1])
            * h**2
            / 6.0
        )
        return result

    def derivative(self, x_eval: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the first derivative of the spline.

        S'(x) = (y_{i+1} - y_i)/h - (3A^2 - 1)*d2y_i*h/6 + (3B^2 - 1)*d2y_{i+1}*h/6
        """
        x_eval = jnp.asarray(x_eval)
        x_clamped = jnp.clip(x_eval, self.x[0], self.x[-1])
        idx = jnp.searchsorted(self.x, x_clamped, side="right") - 1
        idx = jnp.clip(idx, 0, len(self.x) - 2)

        h = self.x[idx + 1] - self.x[idx]
        A = (self.x[idx + 1] - x_clamped) / h
        B = (x_clamped - self.x[idx]) / h

        result = (
            (self.y[idx + 1] - self.y[idx]) / h
            - (3.0 * A**2 - 1.0) * self.d2y[idx] * h / 6.0
            + (3.0 * B**2 - 1.0) * self.d2y[idx + 1] * h / 6.0
        )
        return result

    def derivative2(self, x_eval: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the second derivative of the spline.

        S''(x) = A * d2y_i + B * d2y_{i+1}
        """
        x_eval = jnp.asarray(x_eval)
        x_clamped = jnp.clip(x_eval, self.x[0], self.x[-1])
        idx = jnp.searchsorted(self.x, x_clamped, side="right") - 1
        idx = jnp.clip(idx, 0, len(self.x) - 2)

        h = self.x[idx + 1] - self.x[idx]
        A = (self.x[idx + 1] - x_clamped) / h
        B = (x_clamped - self.x[idx]) / h

        return A * self.d2y[idx] + B * self.d2y[idx + 1]

    # --- JAX pytree registration ---

    def tree_flatten(self):
        children = (self.x, self.y, self.d2y)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.x, obj.y, obj.d2y = children
        return obj


def _compute_natural_spline_coeffs(
    x: Float[Array, "N"], y: Float[Array, "N"]
) -> Float[Array, "N"]:
    """Compute second derivatives for natural cubic spline via Thomas algorithm.

    Natural boundary conditions: d2y[0] = d2y[-1] = 0.

    The tridiagonal system is:
        h_{i-1} * d2y_{i-1} + 2(h_{i-1} + h_i) * d2y_i + h_i * d2y_{i+1} = rhs_i
    where h_i = x_{i+1} - x_i and rhs_i = 6 * [(y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}]

    Uses jax.lax.fori_loop for JIT compatibility.

    Args:
        x: knot positions, shape (N,)
        y: knot values, shape (N,)

    Returns:
        d2y: second derivatives at knots, shape (N,)
    """
    n = x.shape[0]
    h = x[1:] - x[:-1]  # (N-1,)

    # RHS of the tridiagonal system (for interior points 1..N-2)
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])  # (N-2,)

    # Diagonal elements
    diag = 2.0 * (h[:-1] + h[1:])  # (N-2,) main diagonal
    lower = h[:-1]                   # (N-2,) sub-diagonal (lower[i] = h[i])
    upper = h[1:]                    # (N-2,) super-diagonal (upper[i] = h[i+1])

    # Thomas algorithm: forward sweep
    # We modify diag and rhs in-place (via carry in fori_loop)
    def forward_step(i, carry):
        d, r = carry
        w = lower[i] / d[i - 1]
        d = d.at[i].set(d[i] - w * upper[i - 1])
        r = r.at[i].set(r[i] - w * r[i - 1])
        return (d, r)

    diag_mod = diag.copy()
    rhs_mod = rhs.copy()
    diag_mod, rhs_mod = jax.lax.fori_loop(1, n - 2, forward_step, (diag_mod, rhs_mod))

    # Thomas algorithm: back substitution
    d2y_interior = jnp.zeros(n - 2)
    d2y_interior = d2y_interior.at[-1].set(rhs_mod[-1] / diag_mod[-1])

    def backward_step(i, d2y):
        # i goes from 0 to n-4; we process index (n-4-i) in the interior array
        j = n - 4 - i  # counts down from n-4 to 0
        d2y = d2y.at[j].set((rhs_mod[j] - upper[j] * d2y[j + 1]) / diag_mod[j])
        return d2y

    d2y_interior = jax.lax.fori_loop(0, n - 3, backward_step, d2y_interior)

    # Prepend and append zeros for natural boundary conditions
    d2y = jnp.concatenate([jnp.array([0.0]), d2y_interior, jnp.array([0.0])])
    return d2y


def dst2_ortho(x: Float[Array, "N"]) -> Float[Array, "N"]:
    r"""Orthonormal type-II discrete sine transform (DST-II).

    Matches ``scipy.fft.dst(x, type=2, norm="ortho")`` to machine precision.
    Used to build a differentiable de-wiggled (no-wiggle) linear power
    spectrum via the sine-transform IR-resummation split (issue #30's
    ``pk_nw`` class).

    Construction (FFTW RODFT10 identity, see
    http://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html
    and https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html):
    embed the length-``N`` input into a length-``4N`` real array ``w`` via
    odd symmetry (odd-indexed slots hold ``+x``/``-x`` mirrored about the
    midpoint, even slots are zero), take a single complex FFT of ``w``, and
    read the (negated) imaginary part of bins ``1..N``. This is the
    unnormalized DST-II; scipy's "ortho" convention then rescales it so the
    transform matrix is orthonormal: every output element is scaled by
    ``sqrt(1/(2N))``, except the last (``k = N-1``), which gets the extra
    ``1/sqrt(2)`` factor ``sqrt(1/(4N))``. These exact factors were derived
    empirically against ``scipy.fft.dst`` at N=4 and N=8 (see task report)
    and hold at every N tested (verified through N=65536).

    The whole map is built from real scatter/gather, ``jnp.fft.fft``, and
    elementwise real scaling -- all linear operations -- so ``dst2_ortho``
    is itself linear in ``x``. Consequently ``jax.jvp``/``jax.vjp`` through
    it are exact (input-independent Jacobian) rather than merely accurate;
    see ``idst2_ortho`` below, which is obtained as its exact transpose.

    Args:
        x: input samples, shape (N,), real-valued (float32/float64)

    Returns:
        DST-II coefficients, shape (N,), same dtype as ``x``
    """
    n = x.shape[0]
    idx = jnp.arange(n)
    w = jnp.zeros(4 * n, dtype=x.dtype)
    # Odd extension: w[2n+1] = x[n], w[4N-2n-1] = -x[n] (even slots stay 0).
    # unique_indices=True lets JAX transpose the scatter (needed by
    # idst2_ortho's jax.linear_transpose); the two index sets never
    # overlap (all odd, disjoint ranges [1, 2N-1] and [2N+1, 4N-1]).
    w = w.at[2 * idx + 1].set(x, unique_indices=True)
    w = w.at[4 * n - 2 * idx - 1].set(-x, unique_indices=True)
    F = jnp.fft.fft(w)
    y_unnorm = -jnp.imag(F[1 : n + 1])

    # scipy "ortho" scaling: sqrt(1/(2N)) for k=0..N-2, sqrt(1/(4N)) for k=N-1.
    factor = jnp.full((n,), jnp.sqrt(1.0 / (2.0 * n)), dtype=x.dtype)
    factor = factor.at[-1].set(jnp.sqrt(1.0 / (4.0 * n)))
    return y_unnorm * factor


def idst2_ortho(X: Float[Array, "N"]) -> Float[Array, "N"]:
    r"""Orthonormal inverse type-II discrete sine transform (IDST-II / DST-III).

    Matches ``scipy.fft.idst(x, type=2, norm="ortho")`` to machine precision;
    round-trips exactly with :func:`dst2_ortho` (``idst2_ortho(dst2_ortho(x))
    == x``). See ``scipy.fft.idst`` docs
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idst.html)
    and the FFTW RODFT01 identity
    (http://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html),
    which is the DST-III used here (the mirrored/transpose counterpart of
    the RODFT10 DST-II built in :func:`dst2_ortho`).

    Implemented as the exact linear transpose of :func:`dst2_ortho` via
    ``jax.linear_transpose``, rather than a second hand-derived FFT
    construction. This is valid because ``dst2_ortho``'s "ortho" scaling
    makes it an orthonormal (unitary) real transform, whose inverse equals
    its transpose -- exactly the relationship between scipy's DST-II and
    DST-III under ``norm="ortho"``. Because :func:`dst2_ortho` is linear
    (see its docstring), its transpose is well-defined and exact (no
    linearization error), so both ``dst2_ortho`` and ``idst2_ortho`` have
    input-independent Jacobians and forward-mode (``jvp``) / reverse-mode
    (``vjp``) AD through either agree exactly with each other and with the
    closed-form transform matrix.

    Args:
        X: DST-II coefficients, shape (N,), real-valued (float32/float64)

    Returns:
        Reconstructed samples, shape (N,), same dtype as ``X``
    """
    example = jnp.zeros_like(X)
    (y,) = jax.linear_transpose(dst2_ortho, example)(X)
    return y
