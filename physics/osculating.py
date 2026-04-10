#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/osculating.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

G_AU3_Msun_yr2 = 4.0 * math.pi**2


@dataclass
class OrbitalElements:
    """
    Classical osculating orbital elements for a two-body approximation.

    All angles are in radians.
    Units assumed:
      - position: AU
      - velocity: AU/yr
      - masses: Msun
    """
    a: float                  # semi-major axis [AU]
    e: float                  # eccentricity
    i: float                  # inclination [rad]
    omega: float              # argument of periapsis [rad]
    Omega: float              # longitude of ascending node [rad]
    nu: float                 # true anomaly [rad]
    h: float                  # specific angular momentum magnitude
    energy: float             # specific orbital energy


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n == 0.0:
        return np.zeros_like(v, dtype=float)
    return np.asarray(v, dtype=float) / n


def _wrap_angle(x: float) -> float:
    return float((x + 2.0 * math.pi) % (2.0 * math.pi))


def orbital_elements(r: np.ndarray, v: np.ndarray, mu: float) -> OrbitalElements:
    """
    Compute osculating Keplerian orbital elements from state vectors.

    Parameters
    ----------
    r : array_like, shape (2,) or (3,)
        Relative position vector.
    v : array_like, shape (2,) or (3,)
        Relative velocity vector.
    mu : float
        Gravitational parameter G*(m1+m2) in AU^3/yr^2.

    Returns
    -------
    OrbitalElements
        Classical elements. For a purely planar 2D input, the inclination and
        node angles are set to zero.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if r.shape[0] == 2:
        r = np.array([r[0], r[1], 0.0], dtype=float)
    if v.shape[0] == 2:
        v = np.array([v[0], v[1], 0.0], dtype=float)

    rnorm = _norm(r)
    vnorm = _norm(v)
    if rnorm == 0.0:
        raise ValueError("Position vector must be non-zero.")

    # Specific angular momentum
    h_vec = np.cross(r, v)
    h = _norm(h_vec)

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.cross(k_hat, h_vec)
    n = _norm(n_vec)

    # Eccentricity vector
    e_vec = (np.cross(v, h_vec) / mu) - (r / rnorm)
    e = _norm(e_vec)

    # Specific orbital energy
    energy = 0.5 * vnorm**2 - mu / rnorm

    # Semi-major axis
    if abs(energy) < 1e-14:
        a = math.inf
    else:
        a = -mu / (2.0 * energy)

    # Inclination
    if h == 0.0:
        inc = 0.0
    else:
        inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h)))

    # Longitude of ascending node
    if n == 0.0:
        Omega = 0.0
    else:
        Omega = math.atan2(n_vec[1], n_vec[0])

    # Argument of periapsis
    if e < 1e-14 or n == 0.0:
        omega = 0.0
    else:
        cos_omega = np.dot(n_vec, e_vec) / (n * e)
        cos_omega = max(-1.0, min(1.0, float(cos_omega)))
        omega = math.acos(cos_omega)
        if e_vec[2] < 0.0:
            omega = 2.0 * math.pi - omega

    # True anomaly
    if e < 1e-14:
        # Circular orbit: use angle from node/position
        if n == 0.0:
            nu = math.atan2(r[1], r[0])
        else:
            cos_nu = np.dot(n_vec, r) / (n * rnorm)
            cos_nu = max(-1.0, min(1.0, float(cos_nu)))
            nu = math.acos(cos_nu)
            if np.dot(r, v) < 0.0:
                nu = 2.0 * math.pi - nu
    else:
        cos_nu = np.dot(e_vec, r) / (e * rnorm)
        cos_nu = max(-1.0, min(1.0, float(cos_nu)))
        nu = math.acos(cos_nu)
        if np.dot(r, v) < 0.0:
            nu = 2.0 * math.pi - nu

    return OrbitalElements(
        a=float(a),
        e=float(e),
        i=float(inc),
        omega=_wrap_angle(float(omega)),
        Omega=_wrap_angle(float(Omega)),
        nu=_wrap_angle(float(nu)),
        h=float(h),
        energy=float(energy),
    )


def elements_to_state(
    elements: OrbitalElements,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert orbital elements back to position and velocity vectors.

    This is mainly useful for generating initial conditions or testing.
    """
    a = elements.a
    e = elements.e
    i = elements.i
    omega = elements.omega
    Omega = elements.Omega
    nu = elements.nu

    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("Only bound elliptical orbits with positive finite a are supported.")

    p = a * (1.0 - e**2)
    r_pf = np.array(
        [
            p * math.cos(nu) / (1.0 + e * math.cos(nu)),
            p * math.sin(nu) / (1.0 + e * math.cos(nu)),
            0.0,
        ],
        dtype=float,
    )

    v_pf = np.array(
        [
            -math.sqrt(mu / p) * math.sin(nu),
            math.sqrt(mu / p) * (e + math.cos(nu)),
            0.0,
        ],
        dtype=float,
    )

    cO = math.cos(Omega)
    sO = math.sin(Omega)
    ci = math.cos(i)
    si = math.sin(i)
    co = math.cos(omega)
    so = math.sin(omega)

    # Rotation matrix: Rz(Omega) Rx(i) Rz(omega)
    R = np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ],
        dtype=float,
    )

    r = R @ r_pf
    v = R @ v_pf
    return r, v


def osculating_ellipse_points(
    center: np.ndarray,
    elements: OrbitalElements,
    n_points: int = 300,
) -> np.ndarray:
    """
    Generate XY points for the osculating ellipse in the orbital plane.

    Parameters
    ----------
    center : array_like, shape (2,)
        Reference center to add back into the orbit.
    elements : OrbitalElements
        Orbital elements from `orbital_elements`.
    n_points : int
        Number of sample points around the ellipse.

    Returns
    -------
    ndarray, shape (N, 2)
        XY points for plotting. Returns empty array for invalid or unbound orbits.
    """
    center = np.asarray(center, dtype=float)
    if center.shape[0] != 2:
        raise ValueError("center must be a 2D vector")

    if not np.isfinite(elements.a) or elements.a <= 0.0:
        return np.empty((0, 2), dtype=float)
    if not np.isfinite(elements.e) or elements.e >= 1.0:
        return np.empty((0, 2), dtype=float)

    theta = np.linspace(0.0, 2.0 * math.pi, int(n_points))
    p = elements.a * (1.0 - elements.e**2)
    r = p / (1.0 + elements.e * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    c = math.cos(elements.omega)
    s = math.sin(elements.omega)
    rot = np.array([[c, -s], [s, c]], dtype=float)

    pts = np.vstack([x, y]).T @ rot.T
    pts += center[None, :]
    return pts


def elements_summary(elements: OrbitalElements) -> Dict[str, float]:
    """
    Convenience dict for display in Streamlit tables.
    """
    return {
        "a [AU]": elements.a,
        "e": elements.e,
        "i [deg]": math.degrees(elements.i),
        "omega [deg]": math.degrees(elements.omega),
        "Omega [deg]": math.degrees(elements.Omega),
        "nu [deg]": math.degrees(elements.nu),
        "h [AU^2/yr]": elements.h,
        "energy [AU^2/yr^2]": elements.energy,
    }