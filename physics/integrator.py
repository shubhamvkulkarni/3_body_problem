#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/integrator.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

G_AU3_Msun_yr2 = 4.0 * math.pi**2


@dataclass
class BodyState:
    name: str
    mass_msun: float
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def pairwise_accelerations(
    masses: Sequence[float],
    positions: np.ndarray,
    softening: float = 1e-6,
) -> np.ndarray:
    """
    Compute Newtonian pairwise accelerations for an N-body system in 2D.

    Parameters
    ----------
    masses : sequence of float
        Masses in solar masses.
    positions : ndarray, shape (N, 2)
        Positions in AU.
    softening : float
        Plummer-style softening length in AU.

    Returns
    -------
    ndarray, shape (N, 2)
        Accelerations in AU/yr^2.
    """
    masses = np.asarray(masses, dtype=float)
    positions = _as_array(positions)
    n = len(masses)
    acc = np.zeros_like(positions, dtype=float)
    eps2 = float(softening) ** 2

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dr = positions[j] - positions[i]
            r2 = float(np.dot(dr, dr)) + eps2
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += G_AU3_Msun_yr2 * masses[j] * dr * inv_r3

    return acc


def leapfrog_step(
    masses: Sequence[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    softening: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One symplectic kick-drift-kick step.

    This is a good default for qualitative long-term motion.
    """
    masses = np.asarray(masses, dtype=float)
    positions = _as_array(positions)
    velocities = _as_array(velocities)
    dt = float(dt)

    a0 = pairwise_accelerations(masses, positions, softening=softening)
    v_half = velocities + 0.5 * dt * a0
    pos_new = positions + dt * v_half
    a1 = pairwise_accelerations(masses, pos_new, softening=softening)
    vel_new = v_half + 0.5 * dt * a1

    return pos_new, vel_new


def run_simulation(
    masses: Sequence[float],
    positions: np.ndarray,
    velocities: np.ndarray,
    n_steps: int,
    dt: float,
    softening: float = 1e-6,
) -> dict[str, np.ndarray]:
    """
    Run the simulation and return a history dictionary.

    Output keys:
        positions: (n_steps+1, N, 2)
        velocities: (n_steps+1, N, 2)
        t:         (n_steps+1,)
    """
    masses = np.asarray(masses, dtype=float)
    pos = _as_array(positions).copy()
    vel = _as_array(velocities).copy()

    n_steps = int(n_steps)
    dt = float(dt)

    pos_hist = [pos.copy()]
    vel_hist = [vel.copy()]
    t_hist = [0.0]

    for k in range(n_steps):
        pos, vel = leapfrog_step(masses, pos, vel, dt, softening=softening)
        pos_hist.append(pos.copy())
        vel_hist.append(vel.copy())
        t_hist.append((k + 1) * dt)

    return {
        "positions": np.asarray(pos_hist, dtype=float),
        "velocities": np.asarray(vel_hist, dtype=float),
        "t": np.asarray(t_hist, dtype=float),
    }


class NBodySimulator:
    """
    Lightweight simulator wrapper for repeated stepping.
    """

    def __init__(
        self,
        names: Sequence[str],
        masses_msun: Sequence[float],
        positions_au: Sequence[Sequence[float]],
        velocities_auyr: Sequence[Sequence[float]],
        softening: float = 1e-6,
    ) -> None:
        self.names = list(names)
        self.masses = np.asarray(masses_msun, dtype=float)
        self.positions = np.asarray(positions_au, dtype=float)
        self.velocities = np.asarray(velocities_auyr, dtype=float)
        self.softening = float(softening)
        self.time = 0.0

        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.shape[0] != len(self.masses):
            raise ValueError("masses length must match number of bodies")

    def step(self, dt: float) -> None:
        self.positions, self.velocities = leapfrog_step(
            self.masses,
            self.positions,
            self.velocities,
            dt=float(dt),
            softening=self.softening,
        )
        self.time += float(dt)

    def run(self, n_steps: int, dt: float) -> dict[str, np.ndarray]:
        return run_simulation(
            self.masses,
            self.positions,
            self.velocities,
            n_steps=n_steps,
            dt=dt,
            softening=self.softening,
        )

    def state(self) -> dict[str, np.ndarray]:
        return {
            "names": np.asarray(self.names, dtype=object),
            "masses": self.masses.copy(),
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "time": np.array(self.time, dtype=float),
        }