#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/controls.py

from __future__ import annotations

from typing import Dict, Any, List

import streamlit as st


def sidebar_controls() -> Dict[str, Any]:
    """
    Build and return all sidebar UI controls.

    Returns
    -------
    dict
        {
            "star_masses": [m1, m2, m3],
            "planet_mass": float,
            "dt": float,
            "n_steps": int,
            "softening": float,
            "spin": float,
            "show_osculating": bool,
            "run": bool,
            "reset": bool,
        }
    """

    st.sidebar.header("System parameters")

    star1 = st.sidebar.number_input(
        "Star 1 mass [Msun]", min_value=0.01, value=1.0, step=0.1
    )
    star2 = st.sidebar.number_input(
        "Star 2 mass [Msun]", min_value=0.01, value=0.8, step=0.1
    )
    star3 = st.sidebar.number_input(
        "Star 3 mass [Msun]", min_value=0.01, value=0.6, step=0.1
    )

    planet = st.sidebar.number_input(
        "Planet mass [Earth masses]", min_value=0.0, value=1.0, step=1.0
    )

    st.sidebar.header("Simulation controls")

    dt = st.sidebar.number_input(
        "Time step [yr]", min_value=1e-5, value=0.001, step=0.001, format="%.5f"
    )

    n_steps = st.sidebar.number_input(
        "Steps per run", min_value=10, value=500, step=10
    )

    softening = st.sidebar.number_input(
        "Softening length [AU]", min_value=1e-8, value=1e-4, step=1e-4, format="%.6f"
    )

    spin = st.sidebar.slider(
        "Initial spin direction",
        min_value=-1.0,
        max_value=1.0,
        value=1.0,
        step=1.0,
    )

    show_osculating = st.sidebar.checkbox("Show osculating orbits", value=True)

    st.sidebar.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    run_clicked = col1.button("Run")
    reset_clicked = col2.button("Reset")

    return {
        "star_masses": [star1, star2, star3],
        "planet_mass": planet,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "softening": float(softening),
        "spin": float(spin),
        "show_osculating": show_osculating,
        "run": run_clicked,
        "reset": reset_clicked,
    }