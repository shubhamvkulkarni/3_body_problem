#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/__init__.py

"""
Physics package for the triple-star + planet simulator.

This module exposes a clean, high-level API so the Streamlit app (and any
future extensions) can import everything from one place.
"""

# Units
from .units import (
    G_AU3_Msun_yr2,
    M_EARTH_IN_MSUN,
    earth_masses_to_msun,
    msun_to_earth_masses,
)

# Initial conditions
from .init_conditions import (
    InitialBody,
    build_initial_bodies,
    bodies_to_arrays,
    default_positions,
    heuristic_velocities,
)

# Integrator
from .integrator import (
    NBodySimulator,
    run_simulation,
    leapfrog_step,
)

# Osculating elements
from .osculating import (
    OrbitalElements,
    orbital_elements,
    osculating_ellipse_points,
    elements_summary,
)

# High-level simulation
from .simulate import (
    SimulationConfig,
    simulate,
    simulate_one_step,
    run_and_package,
)

__all__ = [
    # units
    "G_AU3_Msun_yr2",
    "M_EARTH_IN_MSUN",
    "earth_masses_to_msun",
    "msun_to_earth_masses",
    # init conditions
    "InitialBody",
    "build_initial_bodies",
    "bodies_to_arrays",
    "default_positions",
    "heuristic_velocities",
    # integrator
    "NBodySimulator",
    "run_simulation",
    "leapfrog_step",
    # osculating
    "OrbitalElements",
    "orbital_elements",
    "osculating_ellipse_points",
    "elements_summary",
    # simulation
    "SimulationConfig",
    "simulate",
    "simulate_one_step",
    "run_and_package",
]