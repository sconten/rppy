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


import rppy
import numpy as np

# Test reflectivity.py


def test_shuey():
    err = 0.005
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = np.array([32])

    exp = 0.151

    Rpp = rppy.reflectivity.shuey(Vp1, Vs1, p1,
                                  Vp2, Vs2, p2,
                                  np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_aki_richards():
    err = 0.05
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = np.array([32])

    exp = 0.15351

    Rpp = rppy.reflectivity.aki_richards(Vp1, Vs1, p1,
                                         Vp2, Vs2, p2,
                                         np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_zoeppritz():
    # Giving Zoeppritz more leeway because I'm checking against the
    # Aki-Richards approximation. Change back to usual 5% once I get good
    # exact values
    err = 0.1
    Vp1 = 3000
    Vp1 = 3000
    Vp2 = 4000
    Vs1 = 1500
    Vs2 = 2000
    p1 = 2000
    p2 = 2200
    theta1 = np.array([32])

    exp = 0.15351

    Rpp = rppy.reflectivity.zoeppritz(Vp1, Vs1, p1,
                                      Vp2, Vs2, p2,
                                      np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_bortfeld():
    err = 0.01
    Vp1 = 3000.
    Vp2 = 4000.
    Vs1 = 1500.
    Vs2 = 2000.
    p1 = 2000.
    p2 = 2200.
    theta1 = np.array([32])

    exp = 0.15469135

    Rpp = rppy.reflectivity.bortfeld(Vp1, Vs1, p1,
                                     Vp2, Vs2, p2,
                                     np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_snell():
    err = 0.01
    vp1 = 2500
    vs1 = 1725
    vp2 = 3800
    vs2 = 1900
    theta1 = np.array([30])

    theta2E = 49.46
    thetas1E = 20.18
    thetas2E = 22.33

    theta2, thetas1, thetas2, p = rppy.reflectivity.snell(vp1, vp2,
                                                          vs1, vs2,
                                                          np.radians(theta1))

    assert np.abs(np.rad2deg(theta2) - theta2E) < err
    assert np.abs(np.rad2deg(thetas1) - thetas1E) < err
    assert np.abs(np.rad2deg(thetas2) - thetas2E) < err


def test_thomsen():
    err = 0.05
    C = np.zeros(shape=(6, 6))
    C[0][0] = 3.1
    C[1][1] = 3.1
    C[2][2] = 2.2
    C[3][3] = 0.9
    C[4][4] = 0.9
    C[0][2] = -0.7
    C[5][5] = 1.2

    eexp = 0.205
    yexp = 0.167
    dexp = -0.288

    e, d, y, dv = rppy.reflectivity.thomsen(C)

    assert np.abs(e - eexp)/eexp < err
    assert np.abs(y - yexp)/yexp < err
    assert np.abs(d - dexp)/dexp < err


def test_Cij():
    err = 0.05
    e = 0.205
    y = 0.167
    d = -0.288
    p = 2.200
    Vp = 1
    Vs = 0.634

    C11 = 3.1
    C33 = 2.2
    C44 = 0.9
    C66 = 1.2
    C13 = -0.7

    C = rppy.reflectivity.Cij(e, d, y, p, Vp, Vs)

    assert np.abs(C[0][0] - C11)/C11 < err
    assert np.abs(C[2][2] - C33)/C33 < err
    assert np.abs(C[3][3] - C44)/C44 < err
    assert np.abs(C[5][5] - C66)/C66 < err
    assert np.abs(C[0][2] - C13)/C13 < err


def test_ruger_vti():
    err = 0.05
    Vp1 = 2900
    Vp2 = 3100
    Vs1 = 1800
    Vs2 = 1850
    p1 = 2180
    p2 = 2200
    theta1 = np.array([20])
    e1 = 0
    d1 = 0
    e2 = 0.1
    d2 = 0.2

    exp = 0.05

    Rpp = rppy.reflectivity.ruger_vti(Vp1, Vs1, p1, e1, d1,
                                      Vp2, Vs2, p2, e2, d2,
                                      np.radians(theta1))

    assert np.abs(Rpp - exp)/exp < err


def test_exact_vti():
    assert 0 == 1


def test_ruger_hti():
    assert 1 == 0


def test_tsvankin_ortho():
    assert 1 == 0


def test_elastic_impedance():
    assert 0 == 1


def test_extended_elastic_impedance():
    assert 0 == 1


# Test media.py
def test_han_eberhart_phillips():
    assert 0 == 1


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
    em = rppy.media.kuster_toksoz(Km, um, Ki, ui, xi, si)
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err

    # Test needle pores
    si = 'needle'
    Kkt_exp = 36.324
    ukt_exp = 42.894
    em = rppy.media.kuster_toksoz(Km, um, Ki, ui, xi, si)
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err

    # Test penny pores
    si = 'penny'
    alpha = 0.01
    Kkt_exp = 21.612
    ukt_exp = 29.323
    em = rppy.media.kuster_toksoz(Km, um, Ki, ui, xi, si, alpha=alpha)
    print(em['K'])
    print(em['u'])
    assert np.abs(Kkt_exp - em['K'])/Kkt_exp < err
    assert np.abs(ukt_exp - em['u'])/ukt_exp < err


def test_hashin_shtrikman():
    err = 0.005
    K = np.array([36, 75, 2.2])
    u = np.array([45., 31., 0.])
    f = np.array([0.584, 0.146, 0.270])

    Kue = 26.9
    Kle = 7.10
    uue = 24.6
    ule = 0

    Ku, Kl, uu, ul = rppy.media.hashin_shtrikman(K, u, f)

    assert np.abs(Ku - Kue)/Kue < err
    assert np.abs(Kl - Kle)/Kue < err
    assert np.abs(uu - uue)/Kue < err
    assert np.abs(ul - ule)/Kue < err


def test_voight_reuss_hill():
    assert 0 == 1


# ========================================
#           Test fluids.py
def test_ciz_shapiro():
    assert 0 == 1


def test_gassmann():
    err = 0.005
    Kfin = 0
    K0 = 36
    Kin = 12
    phi = 0.2

    # Saturate with gas
    Kfout = 0.133
    exp = 12.29
    Kgas = rppy.fluid.gassmann(K0, Kin, Kfin, Kfout, phi)
    assert np.abs(Kgas - exp)/exp < err

    # Saturate with brine
    Kfout = 3.013
    exp = 17.6
    Kbr = rppy.fluid.gassmann(K0, Kin, Kfin, Kfout, phi)
    assert np.abs(Kbr - exp)/exp < err


def test_batzle_wang_brine():
    err = 0.005

    # Test low-pressure, low-temperature brine properties
    T = 25
    P = 5
    S = 30000
    expected_rho = 1.0186679
    expected_Vp = 1535.572

    fluid = rppy.fluid.batzle_wang(P, T, 'brine', S=S)

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

    fluid = rppy.fluid.batzle_wang(P, T, 'oil', G=G, api=api, Rg=Rg)

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

    fluid = rppy.fluid.batzle_wang(P, T, 'gas', G=G)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['K'] - expected_K)/expected_K < err

    # Test high-pressure, high-temperature gas properties
    T = 180
    P = 13
    G = 0.6
    expected_rho = 0.060788613
    expected_K = 25.39253

    fluid = rppy.fluid.batzle_wang(P, T, 'gas', G=G)

    assert np.abs(fluid['rho'] - expected_rho)/expected_rho < err
    assert np.abs(fluid['K'] - expected_K)/expected_K < err


# Test moduli.py
def test_youngs():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = E

    assert np.abs(rppy.moduli.youngs(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.youngs(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.youngs(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.youngs(u=u, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.youngs(u=u, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.youngs(K=K, L=L) - expected)/expected < err


def test_poissons():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = v

    assert np.abs(rppy.moduli.poissons(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.poissons(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.poissons(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.poissons(u=u, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.poissons(u=u, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.poissons(K=K, L=L) - expected)/expected < err


def test_shear():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = u

    assert np.abs(rppy.moduli.shear(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.moduli.shear(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.shear(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.shear(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.shear(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.shear(K=K, L=L) - expected)/expected < err


def test_bulk():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = K

    assert np.abs(rppy.moduli.bulk(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.moduli.bulk(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.bulk(E=E, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.bulk(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.bulk(v=v, L=L) - expected)/expected < err
    assert np.abs(rppy.moduli.bulk(u=u, L=L) - expected)/expected < err


def test_lame():
    err = 0.005
    E = 70
    v = 0.3
    u = 26.92
    K = 58.33
    L = 40.38
    expected = L

    assert np.abs(rppy.moduli.lame(E=E, v=v) - expected)/expected < err
    assert np.abs(rppy.moduli.lame(E=E, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.lame(E=E, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.lame(v=v, u=u) - expected)/expected < err
    assert np.abs(rppy.moduli.lame(v=v, K=K) - expected)/expected < err
    assert np.abs(rppy.moduli.lame(u=u, K=K) - expected)/expected < err


def test_Vp():
    err = 0.05
    rho = 2.65
    E = 95
    v = 0.07
    K = 37
    u = 44
    L = 8
    Vp = 6.008

    assert np.abs(rppy.moduli.Vp(rho, E=E, v=v) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, E=E, K=K) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, E=E, u=u) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, E=E, L=L) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, v=v, K=K) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, v=v, u=u) - Vp)/Vp < err
    # assert np.abs(rppy.moduli.Vp(rho, v=v, L=L) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, K=K, u=u) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, K=K, L=L) - Vp)/Vp < err
    assert np.abs(rppy.moduli.Vp(rho, u=u, L=L) - Vp)/Vp < err


def test_Vs():
    err = 0.005
    rho = 2.65
    E = 95
    v = 0.07
    K = 37
    u = 44
    L = 8
    Vs = 4.075

    assert np.abs(rppy.moduli.Vs(rho, E=E, v=v) - Vs)/Vs < err
    assert np.abs(rppy.moduli.Vs(rho, E=E, K=K) - Vs)/Vs < err
    assert np.abs(rppy.moduli.Vs(rho, E=E, u=u) - Vs)/Vs < err
    assert np.abs(rppy.moduli.Vs(rho, E=E, L=L) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, v=v, K=K) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, v=v, L=L) - Vs)/Vs < err
    # assert np.abs(rppy.Vs(rho, K=K, L=L) - Vs)/Vs < err
    assert np.abs(rppy.moduli.Vs(rho, u=u, L=L) - Vs)/Vs < err


# Test util.py
def test_tuning_wedge():
    err = 0.005
    Rpp = 1
    f0 = 90
    t = 5
    RppT = 1.406
    assert np.abs(rppy.util.tuning_wedge(Rpp, f0, t) - RppT)/RppT < err


def test_ricker():
    err = 0.005
    cf = 90
    t = np.arange(-16, 16, 1)
    w = np.array([-5.166e-08,  -5.394e-07,  -4.753e-06,
                  -3.530e-05,  -2.204e-04,  -1.154e-03,
                  -5.056e-03,  -1.841e-02,  -5.537e-02,
                  -1.359e-01,  -2.675e-01,  -4.061e-01,
                  -4.336e-01,  -2.137e-01,   2.617e-01,
                  7.755e-01,   1.000e+00,   7.755e-01,
                  2.617e-01,  -2.137e-01,  -4.336e-01,
                  -4.061e-01,  -2.675e-01,  -1.359e-01,
                  -5.537e-02,  -1.841e-02,  -5.056e-03,
                  -1.154e-03,  -2.204e-04,  -3.530e-05,
                  -4.753e-06,  -5.394e-07])

    wvlt = rppy.util.ricker(cf, t)

    check = np.asarray(np.abs((wvlt - w)/w))
    assert check.all() < err


def test_ormsby():
    lc = 5
    lf = 10
    hf = 90
    hc = 120
    t = np.arange(-16, 16, 1)/1000
    w = np.array([-16.6253063, -13.9167537, -15.2576779, -25.44130798,
                  -23.70372832, -26.75197885, -28.58752665, -34.43495848,
                  -44.89129349, -20.43200873, -44.37700582, -77.04468147,
                  12.27096796, -51.47511863, -172.89039784, 228.07233786,
                  612.61056745, 228.07233786, -172.89039784, -51.47511863,
                  12.27096796, -77.04468147, -44.37700582, -20.43200873,
                  -44.89129349, -34.43495848, -28.58752665, -26.75197885,
                  -23.70372832, -25.44130798, -15.2576779, -13.9167537])

    wvlt = rppy.util.ormsby(t, lc, lf, hf, hc)

    assert np.allclose(w, wvlt, rtol=1e-05, atol=1e-08)
