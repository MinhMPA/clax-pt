"""Alcock–Paczynski ratios for the CLASS-PT in-loop AP treatment.

Mirrors ``nonlinear_pt.c:1245-1296`` (local CLASS-PT ``09d5531a``; ref §1). For an
output redshift z > 0 with ``AP = Yes`` and fiducial flat-LCDM matter fraction
``Omfid``, CLASS-PT computes, once per z before any loop::

    hfid   = sqrt(Omfid (1+z)^3 + (1 - Omfid) + Omega0_g (1+z)^4)            # 1272
    hnew   = H(z) / (kmsMpc*100*h) = E(z)                                    # 1276
    hratio = hnew / hfid                                                     # 1278
    Dfid   = trapezoid_{j=1..Nz-1} dz / sqrt(Omfid (1+z')^3 + (1-Omfid)
                                             + Omega0_g (1+z)^4), Nz = 2000  # 1280-1284
    Da     = D_A(z) * kmsMpc*100*h * (1+z) = D_M(z) H0                       # 1288
    Dratio = Da / Dfid                                                       # 1291

``kmsMpc = 3.33564095198145e-6`` (``nonlinear_pt.c:1236``) is 1/c in (km/s)^-1, so
``kmsMpc*100*h`` is H0 in Mpc^-1: ``hnew`` is the dimensionless E(z) and ``Da`` is
D_M(z)·H0. Both ratios are therefore h-independent.

z = 0 and AP = No give (1, 1) (``1267-1269``, ``1293-1296``).

Quirks reproduced on purpose (ref §14.1) — do not "fix" them:
  * inside the ``Dfid`` integrand the radiation term is frozen at the OUTPUT z,
    ``Omega0_g (1+z_pk)^4``, not at the integration variable z';
  * ``Omega0_g`` is photons only (no ultra-relativistic species);
  * the fiducial has no massive-neutrino and no dark-energy freedom.

Inputs/outputs: ``ap_ratios`` takes a traced ``BackgroundResult`` plus static
Python floats ``z``, ``omfid`` and returns two 0-d JAX arrays; ``ap_ratios_np``
is the NumPy twin fed with CLASS-independent scalars E(z) and D_M(z)·H0, used to
check the JAX path and to replay CLASS-PT's own H(z), D_A(z).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from clax.background import BackgroundResult

N_DFID = 2000            # nonlinear_pt.c:1233  `int Nz = 2000` (1999 trapezoid panels)
OMFID_DEFAULT = 0.31     # CLASS-PT default `Omfid` (input.c:3879)


def _hfid(zz, z_out, omfid, omega_g, xp):
    """sqrt(Omfid(1+zz)^3 + (1-Omfid) + Omega0_g(1+z_out)^4).

    nonlinear_pt.c:1272 (hfid, where zz == z_out) and 1283 (the Dfid integrand,
    where zz is the integration variable but the radiation term stays frozen at
    z_out — the quirk of ref §14.1). ``xp`` is ``numpy`` or ``jax.numpy``.
    """
    return xp.sqrt(omfid * (1.0 + zz) ** 3 + (1.0 - omfid) + omega_g * (1.0 + z_out) ** 4)


def _dfid(z, omfid, omega_g, xp):
    """Trapezoid of nonlinear_pt.c:1280-1284 on linspace(0, z, N_DFID).

    The C loop runs j = 1 .. Nz-1 over the panels [dz*(j-1), dz*j] with
    dz = z/(Nz-1) (line 1280), i.e. the composite trapezoid rule on the
    N_DFID-point uniform grid spanning [0, z] — the identical sum, vectorised.
    """
    zz = xp.linspace(0.0, z, N_DFID)
    return xp.trapezoid(1.0 / _hfid(zz, z, omfid, omega_g, xp), zz)


def ap_ratios_np(z: float, omfid: float, Omega_g: float, E_z: float, DM_H0: float
                 ) -> tuple[float, float]:
    """NumPy twin of `ap_ratios` from CLASS-independent scalars E(z), D_M(z)*H0.

    Args:
        z: output redshift (nonlinear_pt.c `z_pk[i_z]`).
        omfid: fiducial flat-LCDM matter fraction (`Omfid`).
        Omega_g: photon density fraction today (`pba->Omega0_g`).
        E_z: H(z)/H0, i.e. CLASS-PT's `hnew` (1276).
        DM_H0: D_M(z)*H0, i.e. CLASS-PT's `Da` (1288).

    Returns:
        (hratio, Dratio) as Python floats; (1.0, 1.0) at z <= 0 (1267-1269).
    """
    if z <= 0.0:                                                  # 1267-1269
        return 1.0, 1.0
    hratio = E_z / _hfid(z, z, omfid, Omega_g, np)                # 1278
    Dratio = DM_H0 / _dfid(z, omfid, Omega_g, np)                 # 1291
    return float(hratio), float(Dratio)


def ap_ratios(bg: BackgroundResult, z: float, omfid: float = OMFID_DEFAULT
              ) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """(hratio, Dratio) at static redshift z for fiducial Omfid; `bg` is traced.

    `z` and `omfid` are static Python floats — they control no array shape but
    must not be traced, since the z == 0 branch below is a Python `if` (mirroring
    the C `if (z_pk[i_z] == 0.)`, 1267). `bg` carries the traced cosmology, so
    both returned 0-d arrays are differentiable wrt CosmoParams through
    `background_solve`.

    `bg.H0` is in Mpc^-1 and `bg.tau_of_loga` in Mpc, so `DM_H0` is dimensionless
    exactly like CLASS-PT's `Da`. `conformal_age - tau(z)` is the comoving
    distance only for a flat universe — clax is flat-only, as is CLASS-PT's
    `Dfid`.
    """
    z = float(z)
    if z <= 0.0:                                                  # 1267-1269
        return jnp.ones(()), jnp.ones(())
    loga = -jnp.log1p(z)
    E_z = bg.H_of_loga.evaluate(loga) / bg.H0                     # 1276  hnew = H/H0
    DM_H0 = (bg.conformal_age - bg.tau_of_loga.evaluate(loga)) * bg.H0   # 1288  Da = D_M*H0
    hratio = E_z / _hfid(z, z, omfid, bg.Omega_g, jnp)            # 1278
    Dratio = DM_H0 / _dfid(z, omfid, bg.Omega_g, jnp)             # 1291
    return hratio, Dratio
