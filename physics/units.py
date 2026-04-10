#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# physics/units.py
from __future__ import annotations

import math

# -----------------------------------------------------------------------------
# Canonical units for this project
# -----------------------------------------------------------------------------
# distance: AU
# mass:     solar masses
# time:     years
#
# In these units:
#   G = 4 * pi^2
# -----------------------------------------------------------------------------
G_AU3_Msun_yr2 = 4.0 * math.pi**2

MSUN_TO_KG = 1.98847e30
M_EARTH_TO_KG = 5.9722e24
AU_TO_M = 1.495978707e11
YR_TO_S = 31557600.0

M_EARTH_IN_MSUN = M_EARTH_TO_KG / MSUN_TO_KG
MSUN_IN_M_EARTH = 1.0 / M_EARTH_IN_MSUN


def earth_masses_to_msun(mearth: float) -> float:
    """Convert Earth masses to solar masses."""
    return float(mearth) * M_EARTH_IN_MSUN


def msun_to_earth_masses(msun: float) -> float:
    """Convert solar masses to Earth masses."""
    return float(msun) * MSUN_IN_M_EARTH


def au_to_m(au: float) -> float:
    """Convert AU to meters."""
    return float(au) * AU_TO_M


def m_to_au(m: float) -> float:
    """Convert meters to AU."""
    return float(m) / AU_TO_M


def yr_to_s(yr: float) -> float:
    """Convert years to seconds."""
    return float(yr) * YR_TO_S


def s_to_yr(seconds: float) -> float:
    """Convert seconds to years."""
    return float(seconds) / YR_TO_S