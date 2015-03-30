#!/usr/bin/env python
# -*- coding: utf-8 -*-
#    rppy - a geophysical library for Python
#    Copyright (C) 2015  Sean M. Contenti
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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


def test_youngs():
    err = 0.01
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = E

    assert np.abs(rppy.youngs(v=v, u=u) - expected) < err
    assert np.abs(rppy.youngs(v=v, K=K) - expected) < err
    assert np.abs(rppy.youngs(v=v, L=L) - expected) < err
    assert np.abs(rppy.youngs(u=u, K=K) - expected) < err
    assert np.abs(rppy.youngs(u=u, L=L) - expected) < err
    assert np.abs(rppy.youngs(K=K, L=L) - expected) < err


def test_poissons():
    err = 0.01
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = v

    assert np.abs(rppy.poissons(E=E, u=u) - expected) < err
    assert np.abs(rppy.poissons(E=E, K=K) - expected) < err
    assert np.abs(rppy.poissons(E=E, L=L) - expected) < err
    assert np.abs(rppy.poissons(u=u, K=K) - expected) < err
    assert np.abs(rppy.poissons(u=u, L=L) - expected) < err
    assert np.abs(rppy.poissons(K=K, L=L) - expected) < err


def test_shear():
    err = 0.01
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = u

    assert np.abs(rppy.shear(E=E, v=v) - expected) < err
    assert np.abs(rppy.shear(E=E, K=K) - expected) < err
    assert np.abs(rppy.shear(E=E, L=L) - expected) < err
    assert np.abs(rppy.shear(v=v, K=K) - expected) < err
    assert np.abs(rppy.shear(v=v, L=L) - expected) < err
    assert np.abs(rppy.shear(K=K, L=L) - expected) < err


def test_bulk():
    err = 0.01
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = K

    assert np.abs(rppy.bulk(E=E, v=v) - expected) < err
    assert np.abs(rppy.bulk(E=E, u=u) - expected) < err
    assert np.abs(rppy.bulk(E=E, L=L) - expected) < err
    assert np.abs(rppy.bulk(v=v, u=u) - expected) < err
    assert np.abs(rppy.bulk(v=v, L=L) - expected) < err
    assert np.abs(rppy.bulk(u=u, L=L) - expected) < err


def test_lame():
    err = 0.01
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = L

    assert np.abs(rppy.lame(E=E, v=v) - expected) < err
    assert np.abs(rppy.lame(E=E, u=u) - expected) < err
    assert np.abs(rppy.lame(E=E, K=K) - expected) < err
    assert np.abs(rppy.lame(v=v, u=u) - expected) < err
    assert np.abs(rppy.lame(v=v, K=K) - expected) < err
    assert np.abs(rppy.lame(u=u, K=K) - expected) < err

