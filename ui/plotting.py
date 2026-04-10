#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/plotting.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go


def plot_system(
    names: List[str],
    positions: np.ndarray,
    history: Dict[str, np.ndarray] | None = None,
    osculating_orbits: Dict[str, np.ndarray] | None = None,
    barycenter: np.ndarray | None = None,
) -> go.Figure:
    """
    Create a Plotly figure showing:
    - current positions
    - trajectory history
    - optional osculating orbits
    - optional barycenter

    Parameters
    ----------
    names : list[str]
    positions : (N, 2)
    history : dict[name -> (T, 2)]
    osculating_orbits : dict[name -> (M, 2)]
    barycenter : (2,)

    Returns
    -------
    Plotly Figure
    """

    fig = go.Figure()

    # Consistent colors
    colors = {
        "Star 1": "#d62728",
        "Star 2": "#ff7f0e",
        "Star 3": "#9467bd",
        "Planet": "#1f77b4",
    }

    # --- trajectories ---
    if history is not None:
        for name in names:
            if name not in history:
                continue

            traj = history[name]
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode="lines",
                    line=dict(width=2, color=colors.get(name)),
                    name=f"{name} trail",
                    hoverinfo="skip",
                )
            )

    # --- current positions ---
    for i, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=12, color=colors.get(name)),
                name=name,
            )
        )

    # --- osculating orbits ---
    if osculating_orbits is not None:
        for name, pts in osculating_orbits.items():
            if pts is None or len(pts) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="lines",
                    line=dict(width=1, dash="dot", color=colors.get(name)),
                    name=f"{name} osculating",
                    hoverinfo="skip",
                )
            )

    # --- barycenter ---
    if barycenter is not None:
        fig.add_trace(
            go.Scatter(
                x=[barycenter[0]],
                y=[barycenter[1]],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name="Barycenter",
            )
        )

    # --- layout ---
    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="x [AU]",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(title="y [AU]"),
        legend=dict(orientation="h"),
    )

    return fig