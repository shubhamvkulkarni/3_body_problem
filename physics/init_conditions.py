#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/init_conditions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .units import M_EARTH_IN_MSUN, G_AU3_Msun_yr2


@dataclass
class InitialBody:
    name: str
    mass_msun: float
    position_au: np.ndarray  # shape (2,)
    velocity_auyr: np.ndarray  # shape (2,)


def _vec(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y)], dtype=float)


def center_positions(
    positions: Dict[str, np.ndarray],
    masses_msun: Sequence[float],
    names: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Shift positions so the center of mass is at the origin.
    """
    if names is None:
        names = list(positions.keys())

    masses = np.asarray(masses_msun, dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")

    com = np.zeros(2, dtype=float)
    for i, name in enumerate(names):
        com += masses[i] * np.asarray(positions[name], dtype=float)
    com /= total_mass

    return {name: np.asarray(positions[name], dtype=float) - com for name in names}


def barycenter(
    masses_msun: Sequence[float],
    positions_au: Sequence[Sequence[float]],
    velocities_auyr: Sequence[Sequence[float]] | None = None,
):
    """
    Mass-weighted barycenter. If velocities are provided, returns both
    position and velocity barycenters.
    """
    masses = np.asarray(masses_msun, dtype=float)
    pos = np.asarray(positions_au, dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")

    r_cm = np.sum(masses[:, None] * pos, axis=0) / total_mass

    if velocities_auyr is None:
        return r_cm

    vel = np.asarray(velocities_auyr, dtype=float)
    v_cm = np.sum(masses[:, None] * vel, axis=0) / total_mass
    return r_cm, v_cm


def masses_to_msun(star_masses_msun: Sequence[float], planet_mass_earth: float) -> List[float]:
    """
    Convert [3 star masses in Msun, 1 planet mass in Earth masses] to Msun.
    """
    if len(star_masses_msun) != 3:
        raise ValueError("Expected exactly 3 star masses.")
    masses = [float(m) for m in star_masses_msun]
    masses.append(float(planet_mass_earth) * M_EARTH_IN_MSUN)
    return masses


def default_positions() -> Dict[str, np.ndarray]:
    """
    A simple default 2D layout in AU.
    """
    return {
        "Star 1": _vec(-0.6, 0.0),
        "Star 2": _vec(0.6, 0.0),
        "Star 3": _vec(0.0, 0.55),
        "Planet": _vec(1.2, 0.15),
    }


def default_velocities() -> Dict[str, np.ndarray]:
    """
    Zero velocities by default. The app can overwrite these with an auto-
    generated initial condition scheme.
    """
    return {
        "Star 1": _vec(0.0, 0.0),
        "Star 2": _vec(0.0, 0.0),
        "Star 3": _vec(0.0, 0.0),
        "Planet": _vec(0.0, 0.0),
    }


def heuristic_velocities(
    masses_msun: Sequence[float],
    positions_au: Dict[str, np.ndarray],
    spin: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Generate simple tangential velocities around the system barycenter.

    This is a starting heuristic, not a dynamically exact solution.
    """
    names = list(positions_au.keys())
    masses = np.asarray(masses_msun, dtype=float)
    pos_stack = np.vstack([np.asarray(positions_au[name], dtype=float) for name in names])

    r_cm = barycenter(masses, pos_stack)
    total_mass = float(np.sum(masses))
    out: Dict[str, np.ndarray] = {}

    for i, name in enumerate(names):
        r = np.asarray(positions_au[name], dtype=float) - r_cm
        dist = float(np.linalg.norm(r))
        if dist < 1e-12:
            out[name] = _vec(0.0, 0.0)
            continue

        # Rough circular speed around total enclosed mass.
        # Slight reduction helps avoid immediate instability for the initial guess.
        v_circ = np.sqrt(G_AU3_Msun_yr2 * max(total_mass - 0.5 * masses[i], 1e-12) / dist)
        tangential = np.array([-r[1], r[0]], dtype=float) / dist
        out[name] = spin * 0.6 * v_circ * tangential

    return out


def build_initial_bodies(
    star_masses_msun: Sequence[float],
    planet_mass_earth: float,
    positions: Dict[str, Sequence[float]] | None = None,
    velocities: Dict[str, Sequence[float]] | None = None,
    center_com: bool = True,
    spin: float = 1.0,
) -> List[InitialBody]:
    """
    Build the initial 4-body state from UI inputs.

    Parameters
    ----------
    star_masses_msun:
        Three stellar masses in solar masses.
    planet_mass_earth:
        Planet mass in Earth masses.
    positions:
        Optional dict with keys: Star 1, Star 2, Star 3, Planet.
        Values are [x, y] in AU.
    velocities:
        Optional dict with keys matching positions. If omitted, a heuristic
        tangential velocity field is generated.
    center_com:
        If True, shift initial positions so the system barycenter is at the origin.
    spin:
        +1 or -1 to control the rotation direction of the heuristic velocities.
    """
    masses = masses_to_msun(star_masses_msun, planet_mass_earth)

    names = ["Star 1", "Star 2", "Star 3", "Planet"]

    if positions is None:
        positions = default_positions()

    pos = {name: np.asarray(positions[name], dtype=float) for name in names}

    if center_com:
        pos = center_positions(pos, masses, names=names)

    if velocities is None:
        vel = heuristic_velocities(masses, pos, spin=spin)
    else:
        vel = {name: np.asarray(velocities[name], dtype=float) for name in names}

    bodies = [
        InitialBody(
            name=name,
            mass_msun=float(masses[i]),
            position_au=np.asarray(pos[name], dtype=float),
            velocity_auyr=np.asarray(vel[name], dtype=float),
        )
        for i, name in enumerate(names)
    ]

    return bodies


def bodies_to_arrays(bodies: Sequence[InitialBody]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert bodies into names, masses, positions, velocities arrays.
    """
    names = [b.name for b in bodies]
    masses = np.asarray([b.mass_msun for b in bodies], dtype=float)
    positions = np.asarray([b.position_au for b in bodies], dtype=float)
    velocities = np.asarray([b.velocity_auyr for b in bodies], dtype=float)
    return names, masses, positions, velocities