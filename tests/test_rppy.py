#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_rppy
----------------------------------

Tests for `rppy` module.
"""

from rppy import rppy
import numpy as np


def test_snell():
    err = 0.01
    vp1 = 2500
    vs1 = 1725
    vp2 = 3800
    vs2 = 1900
    theta1 = 30

    theta2E = 49.46
    thetas1E = 20.18
    thetas2E = 22.33

    theta2, thetas1, thetas2, p = rppy.snell(vp1, vp2, vs1, vs2, np.radians(theta1))

    assert np.abs(np.rad2deg(theta2) - theta2E) < err
    assert np.abs(np.rad2deg(thetas1) - thetas1E) < err
    assert np.abs(np.rad2deg(thetas2) - thetas2E) < err
