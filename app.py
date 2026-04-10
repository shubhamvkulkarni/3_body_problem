#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Constants / units
# -----------------------------------------------------------------------------
# Default unit system for the app:
#   distance = AU
#   mass     = solar masses
#   time     = years
# In these units, G = 4 * pi^2.
G_AU3_Msun_yr2 = 4.0 * math.pi**2
M_EARTH_IN_MSUN = 3.00348961491547e-6


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class Body:
    name: str
    mass_msun: float
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _vec(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y)], dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit_perp(v: np.ndarray) -> np.ndarray:
    """Perpendicular unit vector in 2D."""
    n = _norm(v)
    if n == 0:
        return np.array([0.0, 1.0], dtype=float)
    return np.array([-v[1], v[0]], dtype=float) / n


def masses_to_msun(star_masses: List[float], planet_mass_earth: float) -> List[float]:
    return [float(m) for m in star_masses] + [float(planet_mass_earth) * M_EARTH_IN_MSUN]


def default_positions() -> Dict[str, np.ndarray]:
    return {
        "Star 1": _vec(-0.6, 0.0),
        "Star 2": _vec(0.6, 0.0),
        "Star 3": _vec(0.0, 0.55),
        "Planet": _vec(1.2, 0.15),
    }


def center_positions(positions: Dict[str, np.ndarray], masses: List[float]) -> Dict[str, np.ndarray]:
    names = list(positions.keys())
    total_m = sum(masses)
    com = sum(masses[i] * positions[names[i]] for i in range(len(names))) / total_m
    return {k: v - com for k, v in positions.items()}


def barycenter_state(bodies: List[Body]) -> Tuple[np.ndarray, np.ndarray]:
    total_m = sum(b.mass_msun for b in bodies)
    r_cm = sum(b.mass_msun * b.position for b in bodies) / total_m
    v_cm = sum(b.mass_msun * b.velocity for b in bodies) / total_m
    return r_cm, v_cm


def auto_generate_velocities(
    bodies: List[Body],
    spin: float = 1.0,
) -> None:
    """
    Heuristic starting velocities:
    - each body gets a tangential velocity around the system barycenter
    - speed is scaled from the enclosed mass estimate
    This is not an exact 4-body solution; it is a reasonable initialization.
    """
    r_cm, _ = barycenter_state(bodies)
    total_mass = sum(b.mass_msun for b in bodies)

    for b in bodies:
        r = b.position - r_cm
        dist = _norm(r)
        if dist < 1e-9:
            b.velocity = _vec(0.0, 0.0)
            continue

        # Rough circular speed around the total mass, softened a bit for stability.
        enclosed = max(total_mass - 0.5 * b.mass_msun, 1e-8)
        v_circ = math.sqrt(G_AU3_Msun_yr2 * enclosed / dist)

        # Tangential direction; spin controls clockwise/counterclockwise.
        tangential = _unit_perp(r) * spin
        b.velocity = v_circ * 0.6 * tangential


def direct_accelerations(bodies: List[Body], softening: float = 1e-6) -> List[np.ndarray]:
    """
    Direct Newtonian pairwise accelerations in 2D.
    """
    n = len(bodies)
    acc = [np.zeros(2, dtype=float) for _ in range(n)]
    eps2 = softening**2

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dr = bodies[j].position - bodies[i].position
            r2 = float(np.dot(dr, dr)) + eps2
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            acc[i] += G_AU3_Msun_yr2 * bodies[j].mass_msun * dr * inv_r3
    return acc


def leapfrog_step(bodies: List[Body], dt: float, softening: float = 1e-6) -> None:
    """
    Symplectic kick-drift-kick step.
    """
    a0 = direct_accelerations(bodies, softening=softening)
    for i, b in enumerate(bodies):
        b.velocity = b.velocity + 0.5 * dt * a0[i]

    for b in bodies:
        b.position = b.position + dt * b.velocity

    a1 = direct_accelerations(bodies, softening=softening)
    for i, b in enumerate(bodies):
        b.velocity = b.velocity + 0.5 * dt * a1[i]


def compute_history(
    bodies: List[Body],
    n_steps: int,
    dt: float,
    softening: float = 1e-6,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Returns arrays of shape (n_steps + 1, 2) for each body.
    """
    work = [
        Body(name=b.name, mass_msun=b.mass_msun, position=b.position.copy(), velocity=b.velocity.copy())
        for b in bodies
    ]

    position_history = {b.name: [b.position.copy()] for b in work}
    velocity_history = {b.name: [b.velocity.copy()] for b in work}
    times = [0.0]

    for _ in range(n_steps):
        leapfrog_step(work, dt, softening=softening)
        for b in work:
            position_history[b.name].append(b.position.copy())
            velocity_history[b.name].append(b.velocity.copy())
        times.append(times[-1] + dt)

    return (
        {k: np.asarray(v, dtype=float) for k, v in position_history.items()},
        {k: np.asarray(v, dtype=float) for k, v in velocity_history.items()},
        np.asarray(times, dtype=float),
    )


# -----------------------------------------------------------------------------
# Osculating orbit machinery (2D Keplerian approximation about a chosen center)
# -----------------------------------------------------------------------------
def orbital_elements_2d(r: np.ndarray, v: np.ndarray, mu: float) -> Dict[str, float]:
    """
    Approximate osculating elements in the plane for a two-body interpretation.
    Returns:
      a     semi-major axis (AU)
      e     eccentricity
      omega argument of periapsis (rad)
      h     specific angular momentum (z-component magnitude)
    """
    rnorm = _norm(r)
    vnorm = _norm(v)
    if rnorm < 1e-12:
        return {"a": np.nan, "e": np.nan, "omega": np.nan, "h": np.nan}

    # specific angular momentum (z-component)
    h = r[0] * v[1] - r[1] * v[0]

    # eccentricity vector in 2D embedded in 3D
    # e_vec = (v x h)/mu - r_hat
    # In 2D, v x h_z = (h*v_y, -h*v_x)
    e_vec = np.array([h * v[1], -h * v[0]], dtype=float) / mu - r / rnorm
    e = _norm(e_vec)

    energy = 0.5 * vnorm**2 - mu / rnorm
    if abs(energy) < 1e-12:
        a = np.inf
    else:
        a = -mu / (2.0 * energy)

    omega = math.atan2(e_vec[1], e_vec[0]) if e > 1e-12 else math.atan2(r[1], r[0])

    return {"a": float(a), "e": float(e), "omega": float(omega), "h": float(h)}


def osculating_ellipse_points(
    center: np.ndarray,
    elements: Dict[str, float],
    n: int = 300,
) -> np.ndarray:
    """
    Create ellipse points in inertial coordinates from elements.
    """
    a = elements["a"]
    e = elements["e"]
    omega = elements["omega"]

    if not np.isfinite(a) or a <= 0 or not np.isfinite(e) or e >= 1:
        return np.empty((0, 2), dtype=float)

    theta = np.linspace(0, 2.0 * math.pi, n)
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    c = math.cos(omega)
    s = math.sin(omega)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    xy = np.vstack([x, y]).T @ rot.T
    return xy + center


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def make_figure(
    bodies: List[Body],
    history: Dict[str, np.ndarray],
    show_osculating: bool,
    frame_index: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    colors = {
        "Star 1": "#d62728",
        "Star 2": "#ff7f0e",
        "Star 3": "#9467bd",
        "Planet": "#1f77b4",
    }

    if frame_index is None:
        frame_index = len(next(iter(history.values()))) - 1

    # Trajectories and current markers
    for b in bodies:
        traj = history[b.name][: frame_index + 1]
        current = traj[-1]
        fig.add_trace(
            go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode="lines",
                name=f"{b.name} trail",
                line=dict(width=2, color=colors.get(b.name, None)),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[current[0]],
                y=[current[1]],
                mode="markers+text",
                name=b.name,
                text=[b.name],
                textposition="top center",
                marker=dict(size=12, color=colors.get(b.name, None)),
                hovertemplate=f"{b.name}<br>x=%{{x:.3f}} AU<br>y=%{{y:.3f}} AU<extra></extra>",
            )
        )

    # Barycenter
    r_cm, _ = barycenter_state(bodies)
    fig.add_trace(
        go.Scatter(
            x=[r_cm[0]],
            y=[r_cm[1]],
            mode="markers",
            name="Barycenter",
            marker=dict(size=10, symbol="x"),
            hovertemplate="Barycenter<extra></extra>",
        )
    )

    # Osculating ellipses (approximate)
    if show_osculating:
        total_mass = sum(b.mass_msun for b in bodies)
        r_cm, _ = barycenter_state(bodies)

        for b in bodies:
            # Two simple reference choices:
            # - planet uses barycenter of the 3 stars
            # - stars use full-system barycenter
            if b.name == "Planet":
                stars = [x for x in bodies if x.name.startswith("Star")]
                m_stars = sum(x.mass_msun for x in stars)
                r_ref = sum(x.mass_msun * x.position for x in stars) / m_stars
                v_ref = sum(x.mass_msun * x.velocity for x in stars) / m_stars
                mu = G_AU3_Msun_yr2 * (m_stars + b.mass_msun)
            else:
                r_ref = r_cm
                v_ref = np.zeros(2, dtype=float)
                mu = G_AU3_Msun_yr2 * total_mass

            r_rel = b.position - r_ref
            v_rel = b.velocity - v_ref
            elems = orbital_elements_2d(r_rel, v_rel, mu)
            pts = osculating_ellipse_points(r_ref, elems)
            if pts.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="lines",
                        name=f"{b.name} osculating orbit",
                        line=dict(width=1, dash="dot", color=colors.get(b.name, None)),
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(scaleanchor="y", scaleratio=1, title="x [AU]"),
        yaxis=dict(title="y [AU]"),
        legend=dict(orientation="h"),
    )
    return fig


# -----------------------------------------------------------------------------
# Session state / initialization
# -----------------------------------------------------------------------------
def init_state() -> None:
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = None
    if "velocity_history" not in st.session_state:
        st.session_state.velocity_history = None
    if "times" not in st.session_state:
        st.session_state.times = None
    if "bodies" not in st.session_state:
        st.session_state.bodies = None


def build_bodies_from_ui(
    star_masses_msun: List[float],
    planet_mass_earth: float,
    positions: Dict[str, np.ndarray],
    spin: float,
) -> List[Body]:
    masses = masses_to_msun(star_masses_msun, planet_mass_earth)

    names = ["Star 1", "Star 2", "Star 3", "Planet"]
    pos_centered = center_positions(positions, masses)

    bodies = [
        Body(name=names[i], mass_msun=masses[i], position=pos_centered[names[i]].copy(), velocity=_vec(0.0, 0.0))
        for i in range(4)
    ]
    auto_generate_velocities(bodies, spin=spin)
    return bodies


def bodies_at_frame(
    template_bodies: List[Body],
    position_history: Dict[str, np.ndarray],
    velocity_history: Dict[str, np.ndarray],
    frame_index: int,
) -> List[Body]:
    return [
        Body(
            name=b.name,
            mass_msun=b.mass_msun,
            position=position_history[b.name][frame_index].copy(),
            velocity=velocity_history[b.name][frame_index].copy(),
        )
        for b in template_bodies
    ]


def render_state_panel(container, bodies: List[Body], current_time: float) -> None:
    with container.container():
        st.subheader("Current state")
        st.caption(f"Time = {current_time:.4f} yr")

        table_rows = []
        for b in bodies:
            table_rows.append(
                {
                    "Body": b.name,
                    "Mass [Msun]": b.mass_msun,
                    "x [AU]": float(b.position[0]),
                    "y [AU]": float(b.position[1]),
                    "vx [AU/yr]": float(b.velocity[0]),
                    "vy [AU/yr]": float(b.velocity[1]),
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.subheader("Notes")
        st.markdown(
            "- The osculating orbit overlay is a two-body approximation around a chosen reference center.\n"
            "- The simulation uses a leapfrog integrator for the current 4-body system.\n"
            "- The initial velocity field is a heuristic starting configuration."
        )


def position_editor_fallback(positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    st.caption("Set the initial positions for each body.")
    updated = {}
    for name in ["Star 1", "Star 2", "Star 3", "Planet"]:
        c1, c2 = st.columns(2)
        with c1:
            x = st.number_input(f"{name} x [AU]", value=float(positions[name][0]), step=0.05, key=f"{name}_x")
        with c2:
            y = st.number_input(f"{name} y [AU]", value=float(positions[name][1]), step=0.05, key=f"{name}_y")
        updated[name] = _vec(x, y)
    return updated


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Triple-Star + Planet Simulator", layout="wide")
    init_state()

    st.title("Triple-Star + Planet Motion Simulator")
    st.write(
        "A 2D Newtonian 4-body prototype with three stars, one planet, live trajectories, "
        "and approximate osculating orbit overlays."
    )

    with st.sidebar:
        st.header("System parameters")

        star1 = st.number_input("Star 1 mass [Msun]", min_value=0.01, value=1.0, step=0.1)
        star2 = st.number_input("Star 2 mass [Msun]", min_value=0.01, value=0.8, step=0.1)
        star3 = st.number_input("Star 3 mass [Msun]", min_value=0.01, value=0.6, step=0.1)
        planet = st.number_input("Planet mass [Earth masses]", min_value=0.0, value=1.0, step=1.0)

        st.header("Simulation controls")
        dt = st.number_input("Time step [yr]", min_value=1e-5, value=0.001, step=0.001, format="%.5f")
        n_steps = st.number_input("Steps per run", min_value=10, value=250, step=10)
        frame_delay_ms = st.slider("Frame delay [ms]", min_value=10, max_value=200, value=60, step=10)
        spin = st.slider("Initial spin direction", min_value=-1.0, max_value=1.0, value=1.0, step=1.0)
        softening = st.number_input("Softening length [AU]", min_value=1e-8, value=1e-4, step=1e-4, format="%.6f")
        show_osculating = st.checkbox("Show osculating orbits", value=True)
        preserve_positions_on_reset = st.checkbox("Keep edited positions after reset", value=True)

        col_a, col_b = st.columns(2)
        run_clicked = col_a.button("Run / Advance")
        reset_clicked = col_b.button("Reset")

    default_pos = default_positions()

    if "positions" not in st.session_state or reset_clicked:
        if not preserve_positions_on_reset or "positions" not in st.session_state:
            st.session_state.positions = default_pos.copy()
        else:
            st.session_state.positions = st.session_state.positions.copy()

    if reset_clicked or st.session_state.bodies is None:
        st.session_state.bodies = build_bodies_from_ui(
            [star1, star2, star3],
            planet,
            st.session_state.positions,
            spin=spin,
        )
        st.session_state.history = None
        st.session_state.velocity_history = None
        st.session_state.times = None
        st.session_state.initialized = True

    plot_col, state_col = st.columns([2, 1])
    plot_placeholder = plot_col.empty()
    state_placeholder = state_col.empty()

    if run_clicked:
        # Rebuild bodies from the current inputs and run the simulation.
        initial_bodies = build_bodies_from_ui(
            [star1, star2, star3],
            planet,
            st.session_state.positions,
            spin=spin,
        )
        st.session_state.bodies = initial_bodies
        st.session_state.history, st.session_state.velocity_history, st.session_state.times = compute_history(
            initial_bodies,
            n_steps=int(n_steps),
            dt=float(dt),
            softening=float(softening),
        )
        frame_delay_s = float(frame_delay_ms) / 1000.0

        for frame_index in range(len(st.session_state.times)):
            frame_bodies = bodies_at_frame(
                initial_bodies,
                st.session_state.history,
                st.session_state.velocity_history,
                frame_index,
            )
            fig = make_figure(
                frame_bodies,
                st.session_state.history,
                show_osculating=show_osculating,
                frame_index=frame_index,
            )
            plot_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"simulation_plot_{frame_index}",
            )
            render_state_panel(state_placeholder, frame_bodies, st.session_state.times[frame_index])

            if frame_index < len(st.session_state.times) - 1:
                time.sleep(frame_delay_s)

        st.session_state.bodies = bodies_at_frame(
            initial_bodies,
            st.session_state.history,
            st.session_state.velocity_history,
            len(st.session_state.times) - 1,
        )

    # If no simulation has been run yet, show a single-frame "history".
    if st.session_state.history is None:
        bodies = st.session_state.bodies
        st.session_state.history = {b.name: np.vstack([b.position.copy()]) for b in bodies}
        st.session_state.velocity_history = {b.name: np.vstack([b.velocity.copy()]) for b in bodies}
        st.session_state.times = np.asarray([0.0], dtype=float)

    frame_index = len(st.session_state.times) - 1
    bodies = bodies_at_frame(
        st.session_state.bodies,
        st.session_state.history,
        st.session_state.velocity_history,
        frame_index,
    )
    fig = make_figure(
        bodies,
        st.session_state.history,
        show_osculating=show_osculating,
        frame_index=frame_index,
    )
    plot_placeholder.plotly_chart(
        fig,
        use_container_width=True,
        key="simulation_plot_final",
    )
    render_state_panel(state_placeholder, bodies, st.session_state.times[frame_index])

    st.subheader("Initial positions")
    st.session_state.positions = position_editor_fallback(st.session_state.positions)


if __name__ == "__main__":
    main()
