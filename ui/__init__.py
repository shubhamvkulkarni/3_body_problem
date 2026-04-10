#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/__init__.py

"""
UI package for the Streamlit frontend.

This module groups all user-interface related components:
- controls (sidebar inputs, toggles)
- plotting (visualization of system + orbits)
- interactive components (e.g. draggable canvas)

Keeping UI separate from physics ensures clean architecture and easy upgrades.
"""

# Controls
from .controls import (
    sidebar_controls,
)

# Plotting
from .plotting import (
    plot_system,
)

# Interactive components
from .draggable_canvas import (
    position_editor,
)

__all__ = [
    "sidebar_controls",
    "plot_system",
    "position_editor",
]