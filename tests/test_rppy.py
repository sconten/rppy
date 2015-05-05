#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   rppy - a geophysical library for Python
#   Copyright (c) 2014, Sean M. Contenti
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from rppy import rppy
import numpy as np


def test_thomsen():
    err = 0.05
    C = np.zeros(shape=(6, 6))
    C[0][0] = 3.06
    C[1][1] = 0.64
    C[2][2] = 2.22
    C[3][3] = 0.91
    C[0][2] = -0.68
    C[5][5] = 1.21

    eexp = 0.19
    yexp = 0.17

    e, d, y = rppy.thomsen(C)

    assert np.abs(e - eexp)/eexp < err
    assert np.abs(y - yexp)/yexp < err


def test_ruger_vti():
    err = 0.005
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = 32
    e1 = 0
    d1 = 0
    y1 = 0
    e2 = 0.1
    d2 = 0.1
    y2 = 0.1

    exp = 0

    Rpp = rppy.ruger_vti(Vp1, Vs1, p1,
                         e1, d1, y1,
                         Vp2, Vs2, p2,
                         e2, d2, y2,
                         np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


#def test_avoa_hti():
#    assert 1 == 0
#
#
#def test_avoa_ortho():
#    assert 1 == 0


def test_gassmann():
    err = 0.005
    Kfin = 0
    K0 = 36
    Kin = 12
    phi = 0.2

    # Saturate with gas
    Kfout = 0.133
    exp = 12.29
    Kgas = rppy.gassmann(K0, Kin, Kfin, Kfout, phi)
    assert np.abs(Kgas - exp)/exp < err

    # Saturate with brine
    Kfout = 3.013
    exp = 17.6
    Kbr = rppy.gassmann(K0, Kin, Kfin, Kfout, phi)
    assert np.abs(Kbr - exp)/exp < err


def test_kuster_toksoz():
    err = 0.005
    Km = 37
    um = 44
    Ki = 0
    ui = 0
    xi = 0.01

    # Test spherical pores
    si = 'sphere'
    Kkt_exp = 36.4
    ukt_exp = 43.088
    em = rppy.kuster_toksoz(Km, um, Ki, ui, xi, si)
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err

    # Test needle pores
    si = 'needle'
    Kkt_exp = 36.324
    ukt_exp = 42.894
    em = rppy.kuster_toksoz(Km, um, Ki, ui, xi, si)
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err

    # Test penny pores
    si = 'penny'
    alpha = 0.01
    Kkt_exp = 21.612
    ukt_exp = 29.323
    em = rppy.kuster_toksoz(Km, um, Ki, ui, xi, si, alpha=alpha)
    print(em['K'])
    print(em['u'])
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err


def test_tuning_wedge():
    err = 0.005
    Rpp = 1
    f0 = 90
    t = 5
    RppT = 1.406
    assert np.abs(rppy.tuning_wedge(Rpp, f0, t) - RppT)/RppT < err


def test_batzle_wang_brine():
    err = 0.005

    # Test low-pressure, low-temperature brine properties
    T = 25
    P = 5
    S = 30000
    expected_rho = 1.0186679
    expected_Vp = 1535.572

    fluid = rppy.batzle_wang(P, T, 'brine', S=S)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['Vp'] - expected_Vp)/expected_Vp < err


def test_batzle_wang_oil():
    err = 0.005

    # Test low-pressure, low-temperature oil properties
    T = 25
    P = 5
    G = 0.6
    api = 21
    Rg = 7
    expected_rho = 0.9211315
    expected_Vp = 1469.1498

    fluid = rppy.batzle_wang(P, T, 'oil', G=G, api=api, Rg=Rg)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['Vp'] - expected_Vp)/expected_Vp < err


def test_batzle_wang_gas():
    err = 0.005

    # Test low pressure, low temperature gas properties
    T = 15
    P = 3
    G = 0.6
    expected_rho = 0.02332698
    expected_K = 4.264937

    fluid = rppy.batzle_wang(P, T, 'gas', G=G)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['K'] - expected_K)/expected_K < err

    # Test high-pressure, high-temperature gas properties
    T = 180
    P = 13
    G = 0.6
    expected_rho = 0.060788613
    expected_K = 25.39253

    fluid = rppy.batzle_wang(P, T, 'gas', G=G)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['K'] - expected_K)/expected_K < err


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

    theta2, thetas1, thetas2, p = rppy.snell(vp1, vp2,
                                             vs1, vs2,
                                             np.radians(theta1))

    assert np.abs(np.rad2deg(theta2) - theta2E) < err
    assert np.abs(np.rad2deg(thetas1) - thetas1E) < err
    assert np.abs(np.rad2deg(thetas2) - thetas2E) < err


def test_shuey():
    err = 0.005
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = 32

    exp = 0.151

    Rpp = rppy.shuey(Vp1, Vs1, p1, Vp2, Vs2, p2, np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_aki_richards():
    err = 0.05
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = 32

    exp = 0.15351

    Rpp = rppy.aki_richards(Vp1, Vs1, p1, Vp2, Vs2, p2, np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


#def test_zoeppritz():
#    assert 0 == 1


def test_bortfeld():
    err = 0.01
    Vp1 = 3000.
    Vp2 = 4000.
    Vs1 = 1500.
    Vs2 = 2000.
    p1 = 2000.
    p2 = 2200.
    theta1 = 32.

    exp = 0.15469135

    Rpp = rppy.bortfeld(Vp1, Vs1, p1, Vp2, Vs2, p2, np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_hashin_shtrikman():
    err = 0.005
    K = np.array([36, 75, 2.2])
    u = np.array([45., 31., 0.])
    f = np.array([0.584, 0.146, 0.270])

    Kue = 26.9
    Kle = 7.10
    uue = 24.6
    ule = 0

    Ku, Kl, uu, ul = rppy.hashin_shtrikman(K, u, f)

    assert np.abs(Ku - Kue)/Kue < err
    assert np.abs(Kl - Kle)/Kue < err
    assert np.abs(uu - uue)/Kue < err
    assert np.abs(ul - ule)/Kue < err


#def test_voight_reuss_hill():
#    assert 0 == 1


def test_youngs():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = E

    assert np.abs(rppy.youngs(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.youngs(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.youngs(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.youngs(u=u, K=K) - expected)/expected < err
    assert np.abs(rppy.youngs(u=u, L=L) - expected)/expected < err
    assert np.abs(rppy.youngs(K=K, L=L) - expected)/expected < err


def test_poissons():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = v

    assert np.abs(rppy.poissons(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.poissons(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.poissons(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.poissons(u=u, K=K) - expected)/expected < err
    assert np.abs(rppy.poissons(u=u, L=L) - expected)/expected < err
    assert np.abs(rppy.poissons(K=K, L=L) - expected)/expected < err


def test_shear():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = u

    assert np.abs(rppy.shear(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.shear(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.shear(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.shear(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.shear(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.shear(K=K, L=L) - expected)/expected < err


def test_bulk():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = K

    assert np.abs(rppy.bulk(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.bulk(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.bulk(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.bulk(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.bulk(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.bulk(u=u, L=L) - expected)/expected < err


def test_lame():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = L

    assert np.abs(rppy.lame(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.lame(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.lame(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.lame(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.lame(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.lame(u=u, K=K) - expected)/expected < err


def test_Vp():
    err = 0.005
    rho = 2.65
    E = 95
    v = 0.07
    K = 37
    u = 44
    L = 8
    Vp = 6.008

    assert np.abs(rppy.Vp(rho, E=E, v=v) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, E=E, K=K) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, E=E, u=u) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, E=E, L=L) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, v=v, K=K) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, v=v, u=u) - Vp)/Vp < err
    # assert np.abs(rppy.Vp(rho, v=v, L=L) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, K=K, u=u) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, K=K, L=L) - Vp)/Vp < err
    assert np.abs(rppy.Vp(rho, u=u, L=L) - Vp)/Vp < err


def test_Vs():
    err = 0.005
    rho = 2.65
    E = 95
    v = 0.07
    K = 37
    u = 44
    L = 8
    Vs = 4.075

    assert np.abs(rppy.Vs(rho, E=E, v=v) - Vs)/Vs < err
    assert np.abs(rppy.Vs(rho, E=E, K=K) - Vs)/Vs < err
    assert np.abs(rppy.Vs(rho, E=E, u=u) - Vs)/Vs < err
    assert np.abs(rppy.Vs(rho, E=E, L=L) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, v=v, K=K) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, v=v, L=L) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, K=K, L=L) - Vs)/Vs < err
    assert np.abs(rppy.Vs(rho, u=u, L=L) - Vs)/Vs < err
