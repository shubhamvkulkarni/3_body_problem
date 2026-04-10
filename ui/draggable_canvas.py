#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/draggable_canvas.py

from __future__ import annotations

from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _vec(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y)], dtype=float)


def position_editor(
    positions: Dict[str, np.ndarray],
    key: str = "position_editor",
) -> Dict[str, np.ndarray]:
    """
    Interactive position editor.

    CURRENT IMPLEMENTATION (no custom JS component yet):
    - Plotly scatter plot (visual)
    - Numeric inputs (actual editing mechanism)

    Later you can swap this with a true draggable React component
    without changing the API.

    Parameters
    ----------
    positions : dict[name -> (2,)]
    key : str
        Streamlit key namespace

    Returns
    -------
    dict[name -> np.ndarray]
        Updated positions
    """

    st.subheader("Initial Positions")

    names = ["Star 1", "Star 2", "Star 3", "Planet"]

    # --- Plot (visual only for now) ---
    fig = go.Figure()

    colors = {
        "Star 1": "#d62728",
        "Star 2": "#ff7f0e",
        "Star 3": "#9467bd",
        "Planet": "#1f77b4",
    }

    for name in names:
        p = positions[name]
        fig.add_trace(
            go.Scatter(
                x=[p[0]],
                y=[p[1]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=14, color=colors.get(name)),
                name=name,
            )
        )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="x [AU]",
            scaleanchor="y",
            scaleratio=1,
            zeroline=True,
        ),
        yaxis=dict(title="y [AU]", zeroline=True),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Drag-and-drop is not yet enabled. Adjust positions using the inputs below. "
        "This module is designed to be replaced with a custom draggable component later."
    )

    # --- Numeric controls ---
    updated: Dict[str, np.ndarray] = {}

    for name in names:
        col1, col2 = st.columns(2)

        with col1:
            x = st.number_input(
                f"{name} x [AU]",
                value=float(positions[name][0]),
                step=0.05,
                key=f"{key}_{name}_x",
            )

        with col2:
            y = st.number_input(
                f"{name} y [AU]",
                value=float(positions[name][1]),
                step=0.05,
                key=f"{key}_{name}_y",
            )

        updated[name] = _vec(x, y)

    return updated