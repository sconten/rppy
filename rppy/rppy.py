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


import numpy as np
import matplotlib.pyplot as plt


def ricker(f, t):
    """
    Calculates a standard zero-phase Ricker (Mexican Hat) wavelet for a given
    central frequency

    :param t: Time axis of wavelet.
    :param f: Central frequency of Ricker wavelet.
    """
    R = (1 - 2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)
    return(R)


def ormsby(t, f1, f2, f3, f4):
    """
    Calculates a standard zero-phase Ormsby wavelet for a given trapezoidal
    amplitude spectrum

    :param t: Time axis of wavelet.
    :param f1: Low-frequency stop-band.
    :param f2: Low-frequency corner.
    :param f3: High-frequency corner.
    :param f4: High-frequency stop-band.
    """
    O = (((np.pi*f4)**2/(np.pi*f4 - np.pi*f3)*np.sinc(np.pi*f4*t)**2 -
          (np.pi*f3)**2/(np.pi*f4 - np.pi*f3)*np.sinc(np.pi*f3*t)**2) -
         ((np.pi*f2)**2/(np.pi*f2 - np.pi*f1)*np.sinc(np.pi*f2*t)**2 -
          (np.pi*f1)**2/(np.pi*f2 - np.pi*f1)*np.sinc(np.pi*f1*t)**2))
    return(O)


def thomsen(C):
    """
    Returns the Thomsen parameters that characterise transversely isotropic
    materials, computed from the components of the elastic stiffness matrix C.
    """

    e = (C[0][0] - C[2][2]) / (2*C[2][2])
    d = (((C[0][2] + C[3][3])**2 - (C[2][2] - C[3][3])**2) /
         (2*C[2][2]*(C[2][2] - C[3][3])))
    y = (C[5][5] - C[3][3]) / (2*C[3][3])

    return(e, d, y)


def ruger_vti(Vp1, Vs1, p1, e1, d1, y1,
              Vp2, Vs2, p2, e2, d2, y2, theta1):
    """
    Computes the reflectivity response for a weakly anisotripic material with
    vertically transverse isotropy using the equations of Thomsen (1992) and
    Ruger (1997).
    """
    theta2, thetas1, thetas2, p = snell(Vp1, Vp2, Vs1, Vs2, theta1)
    theta = (theta1 + theta2)/2

    u1 = p1*Vs1**2
    u2 = p2*Vs2**2
    Z1 = p1*Vp1
    Z2 = p2*Vp2

    a = (Vp1 + Vp2)/2
    B = (Vs1 / Vs2)/2
    Z = (Z1 + Z2)/2
    u = (u1 + u2)/2

    dZ = Z2 - Z1
    da = Vp2 - Vp1
    du = u2 - u1
    dd = d2 - d1
    de = e2 - e1

    Rpp_iso = ((1/2)*(dZ/Z) +
               (1/2)*(da/a - (2*B/a)**2*(du/u))*np.sin(theta)**2 +
               (1/2)*(da/a)*np.sin(theta)**2*np.tan(theta)**2)

    Rpp_an = (dd/2)*np.sin(theta)**2 + (de/2)*np.sin(theta)**2*np.tan(theta)**2

    Rpp = Rpp_iso + Rpp_an

    return(Rpp)


def exact_vti(V1, V2, V3, V4, p1, p2, theta1,
              C1_11, C1_13, C1_33, C1_55,
              C2_11, C2_13, C2_33, C2_55):
    """
    Returns the exact reflectivity coefficients for a VTI medium computed using
    the relations of Daley and Hron (1977).

    :param V1: P-velocity of upper medium.
    :param V2: P-velocity of lower medium.
    :param V3: S-velocity of upper medium.
    :param V4: S-velocity of lower medium.
    :param p1: Density of upper medium.
    :param p2: Density of lower medium.
    :param theta1: Incidence angle of incident P-wave in upper medium.
    """

    # TODO: There's gotta be a better way to implement these equation than
    #       transcribing them from the 1977 paper. This is insane.
    theta2, theta3, theta4, p = snell(V1, V2, V3, V4, theta1)

    x = np.sin(theta1)
    n = V1/V2
    k1 = V3/V1
    k2 = V4/V2

    P = np.sqrt(1 - x**2)
    Q = np.sqrt(1 - k1**2*x**2)
    S = np.sqrt(1 - x**2/n**2)
    R = np.sqrt(1 - k2**2*x**2/n**2)

    # Upper medium parameters
    A1_11 = C1_11 / p1
    A1_33 = C1_33 / p1
    A1_55 = C1_55 / p1
    A1_13 = C1_13 / p1

    A1_1 = 2*(A1_13 + A1_55)**2 - (A1_33 - A1_55)*(A1_11 + A1_33 - 2*A1_55)
    A1_2 = (A1_11 + A1_33 - 2*A1_55)**2 - 4*(A1_13 + A1_55)**2
    Q1 = np.sqrt((A1_33 - A1_55)**2 +
                 2*A1_1*np.sin(theta1) +
                 A1_2*np.sin(theta1))
    l1 = np.sqrt(((Q1 - A1_33 + A1_55)/np.sin(theta1)**2 +
                 (A1_11 + A1_33 - 2*A1_55))/(2*Q1))
    m1 = np.sqrt(((Q1 - A1_11 + A1_55)/np.cos(theta1)**2 +
                 (A1_11 + A1_33 - 2*A1_55))/(2*Q1))
    l3 = np.sqrt(((Q1 - A1_33 + A1_55)/np.sin(theta3)**2 +
                 (A1_11 + A1_33 - 2*A1_55))/(2*Q1))
    m3 = np.sqrt(((Q1 - A1_11 + A1_55)/np.cos(theta3)**2 +
                 (A1_11 + A1_33 - 2*A1_55))/(2*Q1))

    d1 = l3*C1_33 - m3*C1_13
    e1 = l1*C1_13 + (m1*C1_33 - l1*C1_13)*np.cos(theta1)**2
    w1 = V1/V3*(1/(l1+m1))*(m3*np.cos(theta3)**2 - l3*np.sin(theta3)**2)
    B1 = C1_55

    # Lower medium parameters
    A2_11 = C2_11 / p2
    A2_33 = C2_33 / p2
    A2_55 = C2_55 / p2
    A2_55 = C2_55 / p2
    A2_13 = C2_13 / p2

    A2_1 = 2*(A2_13 + A2_55)**2 - (A2_33 - A2_55)*(A2_11 + A2_33 - 2*A2_55)
    A2_2 = (A2_11 + A2_33 - 2*A2_55)**2 - 4*(A2_13 + A2_55)**2
    Q2 = np.sqrt((A2_33 - A2_55)**2 +
                 2*A2_1*np.sin(theta2) +
                 A2_2*np.sin(theta2))
    l2 = np.sqrt(((Q2 - A2_33 + A2_55)/np.sin(theta2)**2 +
                 (A2_11 + A2_33 - 2*A2_55))/(2*Q2))
    m2 = np.sqrt(((Q2 - A2_11 + A2_55)/np.cos(theta2)**2 +
                 (A2_11 + A2_33 - 2*A2_55))/(2*Q2))
    l4 = np.sqrt(((Q2 - A2_33 + A2_55)/np.sin(theta4)**2 +
                 (A2_11 + A2_33 - 2*A2_55))/(2*Q2))
    m4 = np.sqrt(((Q2 - A2_11 + A2_55)/np.cos(theta4)**2 +
                 (A2_11 + A2_33 - 2*A2_55))/(2*Q2))

    d2 = l4*C2_33 - m4*C2_13
    e2 = V1/V2*(l2*C2_13 + (m2*C2_33 - l2*C2_13)*np.cos(theta2)**2)
    w2 = V1/V4*(1/(l1+m1))*(m4*np.cos(theta4)**2 - l4*np.sin(theta4)**2)
    B2 = C2_55

    l = (l2+m2)/(l1+m1)

    T1 = e2 - (e1*l2)/(n*l1)
    T2 = B2*w2*k1*l3/m1 - B1*(w1*k2*l4)/(n*m1)
    T3 = B2*w2 + B1*(k2*x**2*l4)/(n*m1)
    T4 = e2*m3/l1 + (d1*x**2*l2)/(n*l1)
    T5 = B1*(w1 + k1*x**2*l3/m1)
    T6 = e2*m4/l1 + (d2*x**2*l2)/(n*l1)
    T7 = B2*l - B1*m2/m1
    T8 = d2*m3/l1 - d1*m4/l1
    T9 = B2*(w2*m2/m1 + k2*x**2*l*l4/(n*m1))
    T10 = e1*m3/l1 + d1*x**2
    T11 = e1*m4/l1 + d2*x**2
    T12 = B2*k1*x**2*l*l3/m1 + B1*w1*m2/m1

    E1 = T1*T2*x**2
    E2 = T3*T4*P*Q
    E3 = T5*T6*P*R
    E4 = T7*T8*x**2*P*Q*R*S
    E5 = T9*T10*Q*S
    E6 = T11*T12*R*S
    D = E1 + E2 + E3 + E4 + E5 + E6

    Rpp = (-E1 + E2 + E3 + E4 - E5 - E6)/D

    return(Rpp)


def avoa_hti():
    """
    Calculate P-wave reflection coefficient (Rpp) as a function of incidence
    angle and azimuth for an anisotropic material with horizontal transverse
    isotropy.
    """
    return None


def avoa_ortho():
    return None


def ciz_shapiro(K0, Kdry, Kf, u0, udry, uf, phi, Kphi=None, uphi=None):
    """
    Generalized form of Gassmann's equation to perform fluid substitution to
    allow for a solid (non-zero shear modulus) pore-filling material.

    """
    if Kphi is None:
        Kphi = K0

    Ksat = (1/Kdry - (1/Kdry - 1/K0)**2 /
            (phi*(1/Kf - 1/Kphi) + (1/Kdry - 1/K0)))

    usat = (1/udry - (1/udry - 1/u0)**2 /
            (phi*(1/uf - 1/uphi) + (1/udry - 1/u0)))

    return(Ksat, usat)


def gassmann(K0, Kin, Kfin, Kfout, phi):
    """
    Use Gassmann's equation to perform fluid substitution. Use the bulk modulus
    of a rock saturated with one fluid (or dry frame, Kfin=0) to preduct the
    bulk modulus of a rock second with a second fluid.

    :param K0: Frame mineral modulus (Gpa)
    :param Kin: Input rock modulus (can be fluid saturated or dry)
    :param Kfin: Bulk modulus of the pore-filling fluid of the inital rock
                 (0 if input is the dry-rock modulus)
    :param Kfout: Bulk modulus of the pore-filling fluid of the output
                  (0 if output is dry-rock modulus)
    :param phi: Porosity of the rock
    """
    A = Kfout / (phi*(K0 - Kfout))
    B = Kin / (K0 - Kin)
    C = Kfin / (phi*(K0 - Kfin))
    D = A + B - C

    Kout = K0*D / (1 + D)

    return(Kout)


def kuster_toksoz(Km, um, Ki, ui, xi, si, alpha=None):
    """
    Calculate the effective bulk and shear moduli of a background medium after
    introducing inclusions. Uses Berryman's generalization of the Kuster-
    Toksoz inclusion model.

    Note: Function is not currently set up to natively handle a multiphase
    material with more than one inclusion type.

    :param Km: Bulk modulus of the background medium.
    :param um: Shear modulus of the background medium.
    :param Ki: Bulk modulus of the inclusion material.
    :param ui: Shear modulus of the inclusion material.
    :param xi: Volume fraction of the inclusions
    :param si: Shape of the inclusions - sphere, needle, or, penny
    :param alpha: Aspect ratio of penny crack
    """
    def zeta(K, u):
        Z = u/6*(9*K + 8*u)/(K + 2*u)
        return(Z)

    def gamma(K, u):
        g = u*(3*K + u)/(3*K + 7*u)
        return(g)

    def beta(K, u):
        B = u*(3*K + u)/(3*K + 4*u)
        return(B)

    if si == 'sphere':
        Pmi = (Km + 4/3*um)/(Ki + 4/3*um)
        Qmi = (um + zeta(Km, um))/(ui + zeta(Km, um))
    elif si == 'needle':    # Manually vetted with RPH p.185 parameters
        Pmi = (Km + um + 1/3*ui)/(Ki + um + 1/3*ui)
        Qmi = 1/5*(4*um / (um + ui) +
                   2*(um + gamma(Km, um))/(ui + gamma(Km, um)) +
                   (Ki + 4/3*um)/(Ki + um + 1/3*ui))
    elif si == 'disk':
        Pmi = (Km + 4/3*ui)/(Ki + 4/3*ui)
        print(Pmi)
        if ui > 0:
            Qmi = (um + zeta(Ki, ui))/(ui + zeta(Ki, ui))
        else:
            Qmi = 0
    elif si == 'penny':
        Pmi = (Km + 4/3*ui)/(Ki + 4/3*ui + np.pi*alpha*beta(Km, um))
        Qmi = 1/5*(1 +
                   8*um / (4*ui + np.pi*alpha*(um + 2*beta(Km, um))) +
                   2*(Ki + 2/3*(ui + um)) /
                   (Ki + 4/3*ui + np.pi*alpha*beta(Km, um)))
        print(Pmi)
        print(Qmi)
    Kkt = (((Km + 4/3*um)*Km + 4/3*xi*(Ki - Km)*Pmi*um) /
           (Km + 4/3*um - xi*(Ki - Km)*Pmi))

    ukt = ((xi*(ui - um)*Qmi*zeta(Km, um) + (um + zeta(Km, um))*um) /
           (um + zeta(Km, um) - xi*(ui - um)*Qmi))

    out = {'K': Kkt, 'u': ukt}
    return(out)


def tuning_wedge(Rpp, f0, t):
    """
    Calculate the amplitude at the interface between the top two layers of a
    three layer system. (Intended as an aid to calculate the hypothetical
    thin-bed or tuning-wedge response)

    :param Rpp: Reflection coefficient between the top two layers
    :param f0: Dominant frequency of the Ricker source wavelet
    :param t: Time thickness of layer 2
    """
    A = Rpp*(1 - (1 - 2*np.pi**2*f0**2*(t/1000)**2) *
             np.exp(-np.pi**2*(t/1000)**2*f0**2))

    return A


def batzle_wang(P, T, fluid, S=None, G=None, api=None, Rg=None):
    """
    Calculate the elastic properties of reservoir fluids using the
    Batzle & Wang [1992] equations.

    :param P: Pressure (MPa)
    :param T: Temperature {deg C)
    :param fluid: Fluid type to calculate: brine, gas, or oil
    :param S: Salinity (brine only, in ppm)
    :param G: Gas gravity (gas mode only, ratio of gas density to air density
              at 15.6C and atmospheric pressure)
    :param api: American Petroleum Insitute (API) oil gravity
    :param Rg: Gas-oil ratio
    """

    if fluid == 'brine':
        S = S / (10**6)     # ppm to fraction of one
        w = np.array([
                      [1402.85,   1.524,     3.437e-3,  -1.197e-5],
                      [4.871,    -0.0111,    1.739e-4,  -1.628e-6],
                      [-0.04783,   2.747e-4, -2.135e-6,   1.237e-8],
                      [1.487e-4, -6.503e-7, -1.455e-8,   1.327e-10],
                      [-2.197e-7,  7.987e-10, 5.230e-11, -4.614e-13],
        ])

        rhow = (1 + (10**-6)*(-80*T - 3.3*(T**2) + 0.00175*(T**3) +
                489*P - 2*T*P + 0.016*(T**2)*P - (1.3e-5)*(T**3)*P -
                0.333*(P**2) - 0.002*T*(P**2)))

        rhob = rhow + S*(0.668 + 0.44*S + (10**-6)*(300*P - 2400*P*S +
                         T*(80 + 3*T - 3300*S - 13*P + 47*P*S)))

        Vw = 0
        for i in range(4):
            for j in range(3):
                Vw = Vw + w[i][j]*T**i*P**j

        Vb = (Vw + S*(1170 - 9.8*T + 0.055*T**2 - 8.5e-5*T**3 + 2.6*P -
              0.0029*T*P - 0.0476*P**2) + S**(3/2)*(780 - 10*P + 0.16*P**2) -
              1820*S**2)

        out = {'rho': rhob, 'Vp': Vb}

    elif fluid == 'oil':
        rho0 = 141.5 / (api + 131.5)
        B0 = 0.972 + 0.00038*(2.4*Rg*(G/rho0)**0.5 + T + 17.8)**(1.175)

        rho_r = (rho0/B0)*(1 + 0.001*Rg)**-1    # pseudo-density of oil
        rhog = (rho0 + 0.0012*G*Rg)/B0          # density of oil with gas
        rhop = (rhog + (0.00277*P -             # correct for pressure
                1.71e-7*P**3)*(rhog - 1.15)**2 + 3.49e-4*P)

        rho = rhop / (0.972 + 3.81e-4*(T + 17.78)**1.175)  # correct for temp
        Vp = 2096*(rho_r / (2.6 - rho_r))**0.5 - 3.7*T + 4.64*P

        out = {'rho': rho, 'Vp': Vp}

    elif fluid == 'gas':
        Ta = T + 273.15                 # absolute temperature
        Pr = P / (4.892 - 0.4048*G)     # pseudo-pressure
        Tr = Ta / (94.72 + 170.75*G)    # pseudo-temperature

        R = 8.31441
        d = np.exp(-(0.45 + 8*(0.56 - 1/Tr)**2)*Pr**1.2/Tr)
        c = 0.109*(3.85 - Tr)**2
        b = 0.642*Tr - 0.007*Tr**4 - 0.52
        a = 0.03 + 0.00527*(3.5 - Tr)**3
        m = 1.2*(-(0.45 + 8*(0.56 - 1/Tr)**2)*Pr**0.2/Tr)
        y = (0.85 + 5.6/(Pr + 2) + 27.1/(Pr + 3.5)**2 -
             8.7*np.exp(-0.65*(Pr + 1)))
        f = c*d*m + a
        E = c*d
        Z = a*Pr + b + E

        rhog = (28.8*G*P) / (Z*R*Ta)
        Kg = P*y / (1 - Pr*f/Z)

        out = {'rho': rhog, 'K': Kg}
    else:
        out = None

    return(out)


def snell(vp1, vp2, vs1, vs2, theta1):
    """
    Calculates the angles of and refraction and reflection for an incident
    P-wave in a two-layered system.

    :param vp1: Compressional velocity of upper layer.
    :param vp2: Compressional velocity of lower layer.
    :param vs1: Shear velocity of upper layer.
    :param vs2: Shear velocity of lower layer.
    :param theta1: Angle of incidence of P-wave in upper layer
    """
    theta_crit_1 = np.arcsin(vp1/vp2)
    theta_crit_2 = np.arcsin(vp1/vs2)
    p = np.sin(theta1)/vp1          # Ray parameter
    if theta1 >= theta_crit_1:
        theta2 = None
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = np.arcsin(p*vs2)  # S-wave refraction
    elif theta1 >= theta_crit_2:
        theta2 = None
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = None              # S-wave refraction
    else:
        theta2 = np.arcsin(p*vp2)   # P-wave refraction
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = np.arcsin(p*vs2)  # S-wave refraction

        return(theta2, thetas1, thetas2, p)


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    Calculate the AVO response for a PP reflection based on the Shuey
    (three-term) approximation to the Zoeppritz equations.

    :param vp1: Compressional velocity of upper layer.
    :param vs1: Shear velocity of upper layer.
    :param rho1: Density of upper layer.
    :param vp2: Compressional velocity of lower layer.
    :param vs2: Shear velocity of lower layer.
    :param rho2: Density of lower layer.
    :param theta1: Angle of incidence for P wave in upper layer.
    """
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)
    dvp = vp2 - vp1
    drho = rho2 - rho1
    dvs = vs2 - vs1
    rho = (rho1 + rho2) / 2.
    vs = (vs1 + vs2) / 2.
    vp = (vp1 + vp2) / 2.
    theta = (theta1 + theta2)/2

    A = 0.5*(dvp/vp + drho/rho)
    B = 0.5*dvp/vp - 2*vs**2/vp**2*(drho/rho + 2*dvs/vs)
    C = 0.5*dvp/vp

    Rpp = A + B*np.sin(theta)**2 + C*(np.tan(theta)**2 - np.sin(theta)**2)

    return(Rpp)


def aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    Calculate the AVO response for a PP reflection based on the Aki-Richards
    approximation to the Zoeppritz equations.

    :param vp1: Compressional velocity of upper layer.
    :param vs1: Shear velocity of upper layer.
    :param rho1: Density of upper layer.
    :param vp2: Compressional velocity of lower layer.
    :param vs2: Shear velocity of lower layer.
    :param rho2: Density of lower layer.
    :param theta1: Angle of incidence for P wave in upper layer.
    """
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    theta = (theta1 + theta2) / 2.
    rho = (rho1 + rho2) / 2.
    vp = (vp1 + vp2) / 2.
    vs = (vs1 + vs2) / 2.

    Rpp = (0.5 * (1 - 4 * p**2 * vs**2) * (drho / rho) +
           (dvp / (2 * np.cos(theta)**2 * vp)) - (4 * p**2 * vs**2 * dvs / vs))

    return(Rpp)


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    Calculate the AVO response for a PP reflection based on the exact matrix
    formulation of the Zoeppritz equations.

    :param vp1: Compressional velocity of upper layer.
    :param vs1: Shear velocity of upper layer.
    :param rho1: Density of upper layer.
    :param vp2: Compressional velocity of lower layer.
    :param vs2: Shear velocity of lower layer.
    :param rho2: Density of lower layer.
    :param theta1: Angle of incidence for P wave in upper layer.
    """
    # Need reflection and refraction angles for Zoeppritz
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)

    M = np.array([
        [-np.sin(theta1), -np.cos(thetas1), np.sin(theta2), np.cos(thetas2)],
        [np.cos(theta1), -np.sin(thetas1), np.cos(theta2), -np.sin(thetas2)],
        [2.*rho1*vs1*np.sin(thetas1)*np.cos(theta1),
         rho2*vs1*(1.-2.*np.sin(thetas2)**2),
         2.*rho2*vs2*np.sin(thetas2)*np.cos(theta2),
         rho2*vs2*(1.-2.*np.sin(thetas2)**2)],
        [-rho1*vp1*(1.-2.*np.sin(thetas1)**2),
         rho1*vs1*np.sin(2.*thetas1),
         rho2*vp2*(1.-2.*np.sin(thetas2)**2),
         -rho2*vs2*np.sin(2.*thetas2)]])

    N = np.array([
        [np.sin(theta1), np.cos(thetas1), -np.sin(theta2), -np.cos(thetas2)],
        [np.cos(theta1), -np.sin(thetas1), np.cos(theta2), -np.sin(thetas2)],
        [2.*rho1*vs1*np.sin(thetas1)*np.cos(theta1),
         rho2*vs1*(1.-2.*np.sin(thetas2)**2),
         2.*rho2*vs2*np.sin(thetas2)*np.cos(theta2),
         rho2*vs2*(1.-2.*np.sin(thetas2)**2)],
        [rho1*vp1*(1.-2.*np.sin(thetas1)**2),
         -rho1*vs1*np.sin(2.*thetas1),
         -rho2*vp2*(1.-2.*np.sin(thetas2)**2),
         rho2*vs2*np.sin(2.*thetas2)]])

    Z = np.dot(np.linalg.inv(M), N)

    Rpp = Z[0][0]
    Rps = Z[1][0]
    Tpp = Z[2][0]
    Tps = Z[3][0]

    return(Rpp, Rps, Tpp, Tps)


def bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    Calculate the AVO response for a PP reflection based on the Bortfeld
    approximation to the Zoeppritz equations.

    :param vp1: Compressional velocity of upper layer.
    :param vs1: Shear velocity of upper layer.
    :param rho1: Density of upper layer.
    :param vp2: Compressional velocity of lower layer.
    :param vs2: Shear velocity of lower layer.
    :param rho2: Density of lower layer.
    :param theta1: Angle of incidence for P wave in upper layer.
    """
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)

    Rpp = (0.5 * np.log((vp2 * rho2 * np.cos(theta1)) /
           (vp1 * rho1 * np.cos(theta2))) + (np.sin(theta1)/vp1)**2 * (vs1**2 -
           vs2**2) * (2 + np.log(rho2 / rho1) / np.log(vs2 / vs1)))

    return Rpp


def hashin_shtrikman(K, u, f):
    """
    Compute the Hashin-Shtrikman upper and lower bounds for a
    multi-constituent mixture.

    :param K: Bulk moduli of the individual constituents.
    :param u: Shear moduli of the individual constituents.
    :param f: Volume fraction of the individual constituents.
    """

    def HSlambda(z):
        L = np.sum(f/(K + (4/3)*z))**-1 - (4/3)*z
        return (L)

    def HSgamma(z):
        G = np.sum((f/(u + z)))**(-1) - z
        return (G)

    def HSzeta(K, u):
        z = (u/6)*((9*K+8*u)/(K+2*u))
        return (z)

    K_hi = HSlambda(np.amax(u))
    K_lo = HSlambda(np.amin(u))

    u_hi = HSgamma(HSzeta(np.amax(K), np.amax(u)))
    u_lo = HSgamma(HSzeta(np.amin(K), np.amin(u)))

    return(K_hi, K_lo, u_hi, u_lo)


def voight_reuss_hill(M, f):
    """
    Compute the Voight, Reuss, and Voight-Reuss-Hill averages for a
    multi-constituent mixture.

    :param M: Input moduli of the individual constituents.
    :param f: Volume fractions of the individual constituents.
    """
    v = np.sum(M*f)
    r = 1/np.sum(f/M)
    h = (v + r) / 2.
    return(v, r, h)


def youngs(v=None, u=None, K=None, L=None, Vp=None, Vs=None, rho=None):
    """
    Compute the Young's modulus of a material given sufficient other moduli.

    :param: v: Poisson's ratio (combine with u, K, or L)
    :param u: Shear modulus (combine with v, K, or L)
    :param K: Bulk modulus (combine with v, u, or L)
    :param L: First Lame parameter (combine with v, u, or K)
    :param Vp: Compressional velocity (combine with Vs and rho)
    :param Vs: Shear velocity (combine with Vp and rho)
    :param rho: Density (combine with Vp and Vs)
    """
    if v and u:
        E = 2*u*(1+v)
    elif v and K:
        E = 3*K*(1-2*v)
    elif v and L:
        E = (L*(1+v)*(1-2*v))/(v)
    elif u and K:
        E = (9*K*u)/(3*K+u)
    elif u and L:
        E = u*(3*L+2*u)/(L+u)
    elif K and L:
        E = 9*K*(K-L)/(3*K-L)
    elif Vp and Vs and rho:
        E = rho*Vs**2*(3*Vp**2-4*Vs**2)/(Vp**2-Vs**2)
    else:
        E = None
    return(E)


def poissons(E=None, u=None, K=None, L=None, Vp=None, Vs=None, rho=None):
    """
    Compute the Poisson's modulus of a material given sufficient other moduli.

    :param: E: Young's ratio (combine with u, K, or L)
    :param u: Shear modulus (combine with E, K, or L)
    :param K: Bulk modulus (combine with E, u, or L)
    :param L: First Lame parameter (combine with E, u, or K)
    :param Vp: Compressional velocity (combine with Vs and rho)
    :param Vs: Shear velocity (combine with Vp and rho)
    :param rho: Density (combine with Vp and Vs)
    """
    if E and u:
        v = (E-2*u)/(2*u)
    elif E and K:
        v = (3*K - E)/(6*K)
    elif E and L:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        v = (2*L)/(E+L+R)
    elif u and K:
        v = (3*K-2*u)/(6*K + 2*u)
    elif u and L:
        v = L/(2*(L+u))
    elif K and L:
        v = L/(3*K-L)
    elif Vp and Vs and rho:
        v = (Vp**2 - 2*Vs**2)/(2*(Vp**2-Vs**2))
    else:
        v = None
    return(v)


def shear(E=None, v=None, K=None, L=None, Vp=None, Vs=None, rho=None):
    """
    Compute the shear modulus of a material given sufficient other moduli.

    :param: E: Young's modulus (combine with v, K, or L)
    :param v: Poisson's ratio (combine with E, K, or L)
    :param K: Bulk modulus (combine with E, v, or L)
    :param L: First Lame parameter (combine with E, v, or K)
    :param Vp: Compressional velocity (combine with Vs and rho)
    :param Vs: Shear velocity (combine with Vp and rho)
    :param rho: Density (combine with Vp and Vs)
    """
    if E and v:
        u = E/(2*(1+v))
    elif E and K:
        u = 3*K*E/(9*K-E)
    elif E and L:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        u = (E-3*L+R)/4
    elif v and K:
        u = 3*K*(1-2*v)/(2*(1+v))
    elif v and L:
        u = L*(1-2*v)/(2*v)
    elif K and L:
        u = (3/2)*(K-L)
    elif Vp and Vs and rho:
        u = rho*Vs**2
    else:
        u = None
    return(u)


def bulk(E=None, v=None, u=None, L=None, Vp=None, Vs=None, rho=None):
    """
    Compute the bulk modulus of a material given sufficient other moduli.

    :param: E: Young's modulus (combine with v, u, or L)
    :param v: Poisson's ratio (combine with E, u, or L)
    :param u: shear modulus (combine with E, v, or L)
    :param L: First Lame parameter (combine with E, v, or u)
    :param Vp: Compressional velocity (combine with Vs and rho)
    :param Vs: Shear velocity (combine with Vp and rho)
    :param rho: Density (combine with Vp and Vs)
    """
    if E and v:
        K = E/(3*(1-2*v))
    elif E and u:
        K = E*u/(3*(3*u-E))
    elif E and L:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        K = (E+3*L+R)/6
    elif v and u:
        K = 2*u*(1+v)/(3*(1-2*v))
    elif v and L:
        K = L*(1+v)/(3*v)
    elif u and L:
        K = (3*L+2*u)/3
    elif Vp and Vs and rho:
        K = rho*(Vp**2 - 4*Vs**2/3)
    else:
        K = None
    return(K)


def lame(E=None, v=None, u=None, K=None, Vp=None, Vs=None, rho=None):
    """
    Compute the first Lame's parameter of a material given other moduli.

    :param: E: Young's modulus (combine with v, u, or K)
    :param v: Poisson's ratio (combine with E, u, or K)
    :param u: shear modulus (combine with E, v, or K)
    :param K: Bulk modulus (combine with E, v, or u)
    :param Vp: Compressional velocity (combine with Vs and rho)
    :param Vs: Shear velocity (combine with Vp and rho)
    :param rho: Density (combine with Vp and Vs)
    """
    if E and v:
        L = E*v/((1+v)*(1-2*v))
    elif E and u:
        L = u*(E - 2*u)/(3*u - E)
    elif E and K:
        L = 3*K*(3*K-E)/(9*K-E)
    elif v and u:
        L = 2*u*v/(1-2*v)
    elif v and K:
        L = 3*K*v/(1+v)
    elif u and K:
        L = (3*K-2*u)/3
    elif Vp and Vs and rho:
        L = rho*(Vp**2 - 2*Vs**2)
    else:
        L = None
    return(L)


def Vp(rho, E=None, v=None, u=None, K=None, L=None):
    """
    Compute P-velocity of a material given other moduli.

    :param E: Young's modulus (combine with v, u, or K)
    :param v: Poisson's ratio (combine with E, u, or K)
    :param u: shear modulus (combine with E, v, or K)
    :param K: Bulk modulus (combine with E, v, or u)
    :param L: First Lame parameter (combine with E, v, or u)
    :param rho: Density
    """
    if E and v:
        u = shear(E=E, v=v)
        K = bulk(E=E, v=v)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E and u:
        K = bulk(E=E, u=u)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E and K:
        u = shear(E=E, K=K)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E and L:
        K = bulk(E=E, L=L)
        u = shear(E=E, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v and u:
        K = bulk(v=v, u=u)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v and K:
        u = shear(v=v, K=K)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v and L:
        K = bulk(v=v, L=L)
        u = shear(v=v, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif u and K:
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif u and L:
        K = bulk(u=u, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif K and L:
        u = shear(K=K, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    else:
        Vp = None
    return(Vp)


def Vs(rho, E=None, v=None, u=None, K=None, L=None):
    """
    Compute S-velocity of a material given other moduli.

    :param E: Young's modulus (combine with v, u, or K)
    :param v: Poisson's ratio (combine with E, u, or K)
    :param u: shear modulus (combine with E, v, or K)
    :param K: Bulk modulus (combine with E, v, or u)
    :param L: First Lame parameter (combine with E, v, or u)
    :param rho: Density
    """
    if u:
        Vs = np.sqrt(u/rho)
    elif E and v:
        u = shear(E=E, v=v)
        Vs = np.sqrt(u/rho)
    elif E and K:
        u = shear(E=E, K=K)
        Vs = np.sqrt(u/rho)
    elif E and L:
        u = shear(E=E, L=L)
        Vs = np.sqrt(u/rho)
    elif v and K:
        u = shear(v=v, K=K)
        Vs = np.sqrt(u/rho)
    elif v and L:
        u = shear(v=v, L=L)
        Vs = np.sqrt(u/rho)
    elif K and L:
        u = shear(K=K, L=L)
        Vs = np.sqrt(u/rho)
    else:
        Vs = None
    return(Vs)


def main(*args):

    K = np.array([36, 2.2])
    u = np.array([31, 2.2])

    fw = np.arange(0, 1, 0.01)

    v = np.empty(np.shape(fw))
    r = np.empty(np.shape(fw))
    h = np.empty(np.shape(fw))
    hsu = np.empty(np.shape(fw))
    hsl = np.empty(np.shape(fw))

    for x in np.arange(0, len(fw)):
        v[x], r[x], h[x] = voight_reuss_hill(K, np.array([1-fw[x], fw[x]]))
        returned = hashin_shtrikman(K, u, np.array([1-fw[x], fw[x]]))
        hsu[x] = returned[0]
        hsl[x] = returned[1]

    plt.plot(fw, v, 'k')
    plt.plot(fw, r, 'k')
    plt.plot(fw, hsu, 'k')
    plt.plot(fw, hsl, 'k')

    plt.axis([0, 1, 0, 40])
    plt.show()

    thetas = np.arange(1, 47, 1)
    Rppz = np.empty(np.shape(thetas))
    Rppb = np.empty(np.shape(thetas))
    Rppak = np.empty(np.shape(thetas))
    Rpps = np.empty(np.shape(thetas))
    Rpvti = np.empty(np.shape(thetas))

    plt.figure(2)
    for n in range(np.size(thetas)):
        dummy = zoeppritz(3000, 1500,
                          2000, 4000,
                          2000, 2200,
                          np.radians(thetas[n]))
        Rppz[n] = dummy[0]
        Rppb[n] = bortfeld(3000, 1500,
                           2000, 4000,
                           2000, 2200,
                           np.radians(thetas[n]))
        Rppak[n] = aki_richards(3000, 1500,
                                2000, 4000,
                                2000, 2200,
                                np.radians(thetas[n]))
        Rpps[n] = shuey(3000, 1500, 2000,
                        4000, 2000, 2200,
                        np.radians(thetas[n]))
        Rpvti[n] = ruger_vti(3000, 1500, 2000, 0.0, 0.0, 0.0,
                             4000, 2000, 2200, 0.1, 0.1, 0.1,
                             np.radians(thetas[n]))

    plt.plot(thetas, Rppz, thetas, Rppb, thetas, Rppak, thetas, Rpps, thetas, Rpvti)
    plt.legend(['Zoeppritz', 'Bortfeld', 'Aki-Richards', 'Shuey', 'Ruger VTI'])
    plt.xlim([0, 50])
    plt.ylim([0.15, 0.25])
    plt.show()

    t = np.arange(0, 15, 0.1)

    A = tuning_wedge(1, 90, t)

    plt.figure(3)
    plt.plot(A)

    #########################################
    Km = 37
    um = 44
    Ki = 2.25
    ui = 0
    xi = 0.1
    si = 'sphere'

    em = kuster_toksoz(Km, um, Ki, ui, xi, si)

    print(em['K'])
    print(em['u'])


if __name__ == "__main__":
    main()
