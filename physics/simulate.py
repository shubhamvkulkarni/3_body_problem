#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .integrator import NBodySimulator, leapfrog_step, run_simulation
from .osculating import OrbitalElements, elements_summary, orbital_elements

G_AU3_Msun_yr2 = 4.0 * np.pi**2


@dataclass
class SimulationConfig:
    """
    High-level simulation settings.

    Units:
      - distance: AU
      - mass: Msun
      - time: yr
    """
    dt: float = 1e-3
    n_steps: int = 1000
    softening: float = 1e-6
    compute_elements: bool = True


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def barycenter(masses: Sequence[float], positions: np.ndarray, velocities: Optional[np.ndarray] = None):
    """
    Compute the mass-weighted barycenter. If velocities are supplied, also return barycentric velocity.
    """
    masses = np.asarray(masses, dtype=float)
    positions = _as_array(positions)
    msum = float(np.sum(masses))
    r_cm = np.sum(masses[:, None] * positions, axis=0) / msum

    if velocities is None:
        return r_cm

    velocities = _as_array(velocities)
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / msum
    return r_cm, v_cm


def relative_state(
    body_index: int,
    masses: Sequence[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    reference: str = "barycenter",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (r_rel, v_rel, mu) for an osculating-orbit approximation.

    Parameters
    ----------
    body_index : int
        Index of the target body.
    masses : sequence
        Masses in Msun.
    positions, velocities : ndarray, shape (N, 2)
        Current state.
    reference : str
        - "barycenter": use the full-system barycenter for the reference frame.
        - "others": use the barycenter of all other bodies as the reference center.
          This is often a reasonable choice for a planet around a stellar triple.
    """
    masses = np.asarray(masses, dtype=float)
    positions = _as_array(positions)
    velocities = _as_array(velocities)

    if reference == "barycenter":
        r_ref, v_ref = barycenter(masses, positions, velocities)
        mu = G_AU3_Msun_yr2 * float(np.sum(masses))
    elif reference == "others":
        mask = np.ones(len(masses), dtype=bool)
        mask[body_index] = False
        m_ref = float(np.sum(masses[mask]))
        if m_ref <= 0.0:
            raise ValueError("Reference mass must be positive.")
        r_ref = np.sum(masses[mask, None] * positions[mask], axis=0) / m_ref
        v_ref = np.sum(masses[mask, None] * velocities[mask], axis=0) / m_ref
        mu = G_AU3_Msun_yr2 * (m_ref + float(masses[body_index]))
    else:
        raise ValueError("reference must be 'barycenter' or 'others'")

    r_rel = positions[body_index] - r_ref
    v_rel = velocities[body_index] - v_ref
    return r_rel, v_rel, mu


def simulate(
    masses: Sequence[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    config: SimulationConfig = SimulationConfig(),
) -> Dict[str, np.ndarray]:
    """
    High-level runner that integrates the system and optionally computes
    osculating orbital elements at every saved step.

    Returns
    -------
    dict
        {
            "t": (n_steps+1,),
            "positions": (n_steps+1, N, 2),
            "velocities": (n_steps+1, N, 2),
            "elements": list[dict[str, OrbitalElements]]  # optional
            "summaries": list[dict[str, dict[str, float]]] # optional
        }
    """
    masses = np.asarray(masses, dtype=float)
    positions = _as_array(positions)
    velocities = _as_array(velocities)

    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape.")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions and velocities must have shape (N, 2).")
    if len(masses) != positions.shape[0]:
        raise ValueError("Length of masses must match number of bodies.")

    result = run_simulation(
        masses=masses,
        positions=positions,
        velocities=velocities,
        n_steps=int(config.n_steps),
        dt=float(config.dt),
        softening=float(config.softening),
    )

    if not config.compute_elements:
        return result

    pos_hist = result["positions"]
    vel_hist = result["velocities"]
    n_time, n_bodies = pos_hist.shape[0], pos_hist.shape[1]

    elements_hist: List[Dict[int, OrbitalElements]] = []
    summaries_hist: List[Dict[int, Dict[str, float]]] = []

    # By default:
    # - for the planet (assumed last body), use the barycenter of the other bodies
    # - for the stars, use the full-system barycenter
    for k in range(n_time):
        pos_k = pos_hist[k]
        vel_k = vel_hist[k]

        elems_k: Dict[int, OrbitalElements] = {}
        sums_k: Dict[int, Dict[str, float]] = {}

        for i in range(n_bodies):
            ref_mode = "others" if i == n_bodies - 1 else "barycenter"
            r_rel, v_rel, mu = relative_state(i, masses, pos_k, vel_k, reference=ref_mode)
            elems = orbital_elements(r_rel, v_rel, mu)
            elems_k[i] = elems
            sums_k[i] = elements_summary(elems)

        elements_hist.append(elems_k)
        summaries_hist.append(sums_k)

    result["elements"] = elements_hist
    result["summaries"] = summaries_hist
    return result


def simulate_one_step(
    masses: Sequence[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    softening: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for a single step.
    """
    return leapfrog_step(
        masses=np.asarray(masses, dtype=float),
        positions=_as_array(positions),
        velocities=_as_array(velocities),
        dt=float(dt),
        softening=float(softening),
    )


def make_simulator(
    names: Sequence[str],
    masses_msun: Sequence[float],
    positions_au: Sequence[Sequence[float]],
    velocities_auyr: Sequence[Sequence[float]],
    softening: float = 1e-6,
) -> NBodySimulator:
    """
    Convenience factory for the mutable simulator object.
    """
    return NBodySimulator(
        names=names,
        masses_msun=masses_msun,
        positions_au=positions_au,
        velocities_auyr=velocities_auyr,
        softening=softening,
    )


def run_and_package(
    names: Sequence[str],
    masses_msun: Sequence[float],
    positions_au: Sequence[Sequence[float]],
    velocities_auyr: Sequence[Sequence[float]],
    config: SimulationConfig = SimulationConfig(),
) -> Dict[str, object]:
    """
    Run a simulation and return a UI-friendly package with named bodies.

    Output keys
    ----------
    names : ndarray[str]
    history : dict with positions/velocities/time
    elements : list of per-step orbital elements by body name (optional)
    summaries : list of per-step element summaries by body name (optional)
    """
    names = list(names)
    sim = make_simulator(
        names=names,
        masses_msun=masses_msun,
        positions_au=positions_au,
        velocities_auyr=velocities_auyr,
        softening=config.softening,
    )

    result = simulate(
        masses=sim.masses,
        positions=sim.positions,
        velocities=sim.velocities,
        config=config,
    )

    package: Dict[str, object] = {
        "names": np.asarray(names, dtype=object),
        "history": {
            "t": result["t"],
            "positions": result["positions"],
            "velocities": result["velocities"],
        },
    }

    if config.compute_elements:
        named_elements: List[Dict[str, OrbitalElements]] = []
        named_summaries: List[Dict[str, Dict[str, float]]] = []

        for step_elems, step_summaries in zip(result["elements"], result["summaries"]):
            e_by_name: Dict[str, OrbitalElements] = {}
            s_by_name: Dict[str, Dict[str, float]] = {}
            for i, name in enumerate(names):
                e_by_name[name] = step_elems[i]
                s_by_name[name] = step_summaries[i]
            named_elements.append(e_by_name)
            named_summaries.append(s_by_name)

        package["elements"] = named_elements
        package["summaries"] = named_summaries

    return package