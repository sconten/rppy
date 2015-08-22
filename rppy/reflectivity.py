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
    theta1 = np.radians(theta1)
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
    theta1 = np.radians(theta1)
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
    theta1 = np.radians(theta1)
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)

    Rpp = np.zeros(np.shape(theta1))
    Rps = np.zeros(np.shape(theta1))
    Tpp = np.zeros(np.shape(theta1))
    Tps = np.zeros(np.shape(theta1))

    for n in np.arange(0, len(theta1)):

        M = np.array([
            [-np.sin(theta1[n]), -np.cos(thetas1[n]),
             np.sin(theta2[n]), np.cos(thetas2[n])],
            [np.cos(theta1[n]), -np.sin(thetas1[n]),
             np.cos(theta2[n]), -np.sin(thetas2[n])],
            [2.*rho1*vs1*np.sin(thetas1[n])*np.cos(theta1[n]),
             rho2*vs1*(1.-2.*np.sin(thetas2[n])**2),
             2.*rho2*vs2*np.sin(thetas2[n])*np.cos(theta2[n]),
             rho2*vs2*(1.-2.*np.sin(thetas2[n])**2)],
            [-rho1*vp1*(1.-2.*np.sin(thetas1[n])**2),
             rho1*vs1*np.sin(2.*thetas1[n]),
             rho2*vp2*(1.-2.*np.sin(thetas2[n])**2),
             -rho2*vs2*np.sin(2.*thetas2[n])]])

        N = np.array([
            [np.sin(theta1[n]), np.cos(thetas1[n]),
             -np.sin(theta2[n]), -np.cos(thetas2[n])],
            [np.cos(theta1[n]), -np.sin(thetas1[n]),
             np.cos(theta2[n]), -np.sin(thetas2[n])],
            [2.*rho1*vs1*np.sin(thetas1[n])*np.cos(theta1[n]),
             rho2*vs1*(1.-2.*np.sin(thetas2[n])**2),
             2.*rho2*vs2*np.sin(thetas2[n])*np.cos(theta2[n]),
             rho2*vs2*(1.-2.*np.sin(thetas2[n])**2)],
            [rho1*vp1*(1.-2.*np.sin(thetas1[n])**2),
             -rho1*vs1*np.sin(2.*thetas1[n]),
             -rho2*vp2*(1.-2.*np.sin(thetas2[n])**2),
             rho2*vs2*np.sin(2.*thetas2[n])]])

        Z = np.dot(np.linalg.inv(M), N)

        Rpp[n] = Z[0][0]
        Rps[n] = Z[1][0]
        Tpp[n] = Z[2][0]
        Tps[n] = Z[3][0]

    return(Rpp)


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
    theta1 = np.radians(theta1)
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)

    Rpp = (0.5 * np.log((vp2 * rho2 * np.cos(theta1)) /
           (vp1 * rho1 * np.cos(theta2))) + (np.sin(theta1)/vp1)**2 * (vs1**2 -
           vs2**2) * (2 + np.log(rho2 / rho1) / np.log(vs2 / vs1)))

    return Rpp


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

    p = np.full(np.shape(theta1), None)
    theta2 = np.full(np.shape(theta1), None)
    thetas1 = np.full(np.shape(theta1), None)
    thetas2 = np.full(np.shape(theta1), None)

    p = np.sin(theta1)/vp1        # Ray parameter
    thetas1 = np.arcsin(p*vs1)    # S-wave reflection

    # P refraction below first critical angle
    theta2 = np.arcsin(p*vp2)

    # S refraction below second critical angle
    thetas2 = np.arcsin(p*vs2)

    return(theta2, thetas1, thetas2, p)


def thomsen(C, p):
    """
    Returns the Thomsen parameters that characterise orthorhombically isotropic
    materials, computed from the components of the elastic stiffness matrix C.

    Returns Tsvankin's formulation of the parameters extended to HTI and
    orthorhombic symmetries. In the VTI case, e1, d1, and y1 collapse to the
    classical Thomsen formulation, while in the HTI case, e2, d2, and y2
    correspond to the coefficients introduced by Tsvankin [1997]
    and Ruger [1997].

    VP0 — the vertical velocity of the P-wave;
    VS0 — the vertical velocity of the S-wave polarized in the x1-
          direction;

    e(2) — the VTI parameter e in the symmetry plane [x1, x3]
           normal to the x2-axis (close to the fractional difference
           between the P-wave velocities in the x1- and x3-
           directions);
    δ(2) — the VTI parameter δ in the [x1, x3] plane (responsible
           for near-vertical P-wave velocity variations, also
           influences SV-wave velocity anisotropy);
    γ(2) — the VTI parameter γ in the [x1, x3] plane (close to the
           fractional difference between the SH-wave velocities
           in the x1- and x3-directions);
    e(1) — the VTI parameter e in the [x2, x3] plane;
    δ(1) — the VTI parameter δ in the [x2, x3] plane;
    γ(1) — the VTI parameter γ in the [x2, x3] plane;
    δ(3) — the VTI parameter δ in the [x1, x2] plane (x1 is used as
           the symmetry axis).
    """

    e2 = (C[0][0] - C[2][2]) / (2*C[2][2])
    y2 = (C[5][5] - C[3][3]) / (2*C[3][3])
    d2 = (((C[0][2] + C[4][4])**2 - (C[2][2] - C[4][4])**2) /
          (2*C[2][2]*(C[2][2] - C[4][4])))

    e1 = (C[1][1] - C[2][2]) / (2*C[2][2])
    y1 = (C[5][5] - C[4][4]) / (2*C[4][4])
    d1 = (((C[1][2] + C[3][3])**2 - (C[2][2] - C[3][3])**2) /
          (2*C[2][2]*(C[2][2] - C[3][3])))

    d3 = (((C[0][1] + C[5][5])**2 - (C[0][0] - C[5][5])**2) /
          (2*C[0][0]*(C[0][0] - C[5][5])))

    vp = np.sqrt(C[2][2]/p)
    vs = np.sqrt(C[4][4]/p)

    return(vp, vs, e1, d1, y1, e2, d2, y2, d3)


def Cij(Vp, Vs, p, e1, d1, y1, e2, d2, y2, d3):
    """
    Returns the elastic stiffness elements C11, C33, C13, C55, and C66 that
    characterize transversely isotropic materials, using the Thomsen parameters
    and elastic parameters.
    """
    C = np.zeros(shape=(6, 6))

    # On-diagonal components
    C[2][2] = p*Vp**2
    C[4][4] = p*Vs**2
    C[0][0] = C[2][2]*(2*e2 + 1)
    C[1][1] = C[2][2]*(2*e1 + 1)
    C[5][5] = C[4][4]*(2*y1 + 1)
    C[3][3] = C[5][5]*(2*y2 + 1)

    # Off-diagonal
    C[0][2] = np.sqrt((C[2][2] - C[4][4])**2 + 2*C[2][2]*(C[2][2] - C[4][4])*d2) - C[4][4]
    C[1][2] = np.sqrt((C[2][2] - C[3][3])**2 + 2*C[2][2]*(C[2][2] - C[3][3])*d1) - C[3][3]
    C[0][1] = np.sqrt((C[0][0] - C[5][5])**2 + 2*C[0][0]*(C[0][0] - C[5][5])*d3) - C[5][5]

    # Exploit symmetry to fill out matrix
    C[2][0] = C[0][2]
    C[2][1] = C[1][2]
    C[1][0] = C[0][1]

    return(C)


def ruger_vti(Vp1, Vs1, p1, e1, d1,
              Vp2, Vs2, p2, e2, d2, theta1):
    """
    Computes the reflectivity response for a weakly anisotripic material with
    vertically transverse isotropy using the equations of Thomsen (1992) and
    Ruger (1997).
    """
    theta1 = np.radians(theta1)
    theta2, thetas1, thetas2, p = snell(Vp1, Vp2, Vs1, Vs2, theta1)
    theta = (theta1 + theta2)/2
    theta = theta1

    u1 = p1*Vs1**2
    u2 = p2*Vs2**2
    Z1 = p1*Vp1
    Z2 = p2*Vp2

    a = (Vp1 + Vp2)/2
    B = (Vs1 + Vs2)/2
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


def daley_hron_vti(V1, V2, V3, V4, p1, p2, theta1,
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
    theta1 = np.radians(theta1)
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


def ruger_hti(Vp1, Vs1, p1, ev1, dv1, y1,
              Vp2, Vs2, p2, ev2, dv2, y2,
              theta1, phi):
    """
    Calculate P-wave reflection coefficient (Rpp) as a function of incidence
    angle and azimuth for an anisotropic material with horizontal transverse
    isotropy using the Ruger [1996] approximation.
    """
    theta1 = np.radians(theta1)
    phi = np.radians(phi)

    theta2, thetas1, thetas2, p = snell(Vp1, Vp2, Vs1, Vs2, theta1)
    theta = (theta1 + theta2)/2.0
    theta = theta1
    u1 = p1*(Vs1**2)
    u2 = p2*(Vs2**2)
    Z1 = p1*Vp1
    Z2 = p2*Vp2

    a = (Vp1 + Vp2)/2.0
    B = (Vs1 + Vs2)/2.0
    Z = (Z1 + Z2)/2.0
    u = (u1 + u2)/2.0

    dZ = Z2 - Z1
    da = Vp2 - Vp1
    du = u2 - u1
    ddv = dv2 - dv1
    dev = ev2 - ev1
    dy = y2 - y1

    A = (1./2.)*(dZ/Z)

    B = ((1./2.)*((da/a) -
         ((2.*B/a)**2.)*(du/u) +
         (ddv + 2.*((2.*B/a)**2.)*dy)*(np.cos(phi)**2.)))

    C = ((1./2.)*((da/a) +
         dev*(np.cos(phi)**4.) +
         ddv*(np.sin(phi)**2.)*(np.cos(phi)**2.)))

    Rpp = A + B*np.sin(theta)**2 + C*np.sin(theta)**2*np.tan(theta)**2

    return(Rpp)


def elastic_impedance(Vp, Vs, p, theta):
    """
    Calculate the incidence-angle dependent elastic impedance of
    a material. Uses the Aki and Richards approximation for
    isotropic reflection coefficients.
    """
    K = (Vs/Vp)**2
    Ie = (Vp**(1 + np.tan(theta)**2) *
          p**(1 - 4*K*np.sin(theta)**2) *
          Vs**(-8*K*np.sin(theta)**2))

    return(Ie)


def extended_elastic_impedance(Vp, Vs, p, chi, Vp0=1, Vs0=1, p0=1):
    """
    Calculate the Extended Elastic Impedance of Whitcombe [2002] of a material.
    """
    K = (Vs/Vp)**2
    Iee = ((Vp0 * p0) *
           (Vp / Vp0)**(np.cos(chi) + np.sin(chi)) *
           (p / p0)**(np.cos(chi) - 4*K*np.sin(chi)) *
           (Vs / Vs0)**(-8*K*np.sin(chi)))

    return(Iee)


def exact_ortho(C1, p1, C2, p2, chi1, chi2, phi, theta):
    """
    Calculate the exact Zoeppritz equations for an HTI medium using the
    Schoenberg and Protazio [1992] formulation.
    """

    phi = np.radians(phi)
    theta = np.radians(theta)
    chi1 = np.radians(chi1)
    chi2 = np.radians(chi2)

    # Black magic begins here...
    # Adding a very small percentage of whitening to the input matrices seems
    # to stabilize the solution???
    norm1 = np.linalg.norm(C1)
    norm2 = np.linalg.norm(C2)
    C1 = C1 + 0.00001*norm1*np.random.rand(6, 6)
    C2 = C2 + 0.00001*norm2*np.random.rand(6, 6)
    # End black magic

    # Construct rotation matrices to properly align the
    # HTI porion of the orthorhombic anisotropy.
    schi = np.sin(chi1)
    cchi = np.cos(chi1)
    G1 = [[cchi**2, schi**2, 0, 0, 0, 2*cchi*schi],
          [schi**2, cchi**2, 0, 0, 0, -2*schi*cchi],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 0, cchi, -schi, 0],
          [0, 0, 0, schi, cchi,  0],
          [-cchi*schi, cchi*schi, 0, 0, 0, cchi**2 - schi**2]]
    G1 = np.asarray(G1)

    schi = np.sin(chi2)
    cchi = np.cos(chi2)
    G2 = np.zeros(shape=(6, 6))
    G2 = [[cchi**2, schi**2, 0, 0, 0, 2*cchi*schi],
          [schi**2, cchi**2, 0, 0, 0, 2*schi*cchi],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 0, cchi, -schi, 0],
          [0, 0, 0, schi, cchi,  0],
          [-cchi*schi, cchi*schi, 0, 0, 0, cchi**2 - schi**2]]
    G2 = np.asarray(G2)

    # Rotate stiffness matrices
    C1 = G1.dot(C1).dot(G1.T)
    C2 = G2.dot(C2).dot(G2.T)

    ########################################
    # SLOWNESS VECTOR OF THE INCIDENT WAVE
    ########################################

    # Construct Christoffel matrices in the velocity form as the first step
    # towards determining the phase velocity and phase polarization vectors
    # in the direction of propagation
    # (that is, the slowness vector of the incident phase).

    # Propagation vector (directional, no velocity information)
    n = np.array([np.cos(phi)*np.sin(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(theta)])

    # Construct Christoffel matrix
    L = christoffel(C1, n)
    # Compute eigenvectors and eigenvalues of Christoffel matrix.
    w, v = np.linalg.eig(L/p1)
    # quasi-P velocity of the upper medium, in the direction of propagation.
    vp1 = np.sqrt(np.max(w))
    # Slowness vector using derived quasi-P velocity.
    s = n / vp1

    ########################################
    # SLOWNESS VECTORS OF REFLECTED WAVES
    ########################################

    # Compute the coefficients of the bicubic equation from the stiffness
    # matrix of the upper layer, density, and the two horizontal components
    # of slowness.
    A, B, C, D = monoclinic_bicubic_coeffs(s[0], s[1], p1, C1)
    # Input the computed coefficients and solve the bicubic polynomial
    z = np.sort(np.roots(np.array([A, B, C, D])))
    s1P = np.array([s[0], s[1], np.sqrt(np.abs(z[0]))])
    s1S = np.array([s[0], s[1], np.sqrt(np.abs(z[1]))])
    s1T = np.array([s[0], s[1], np.sqrt(np.abs(z[2]))])

    ########################################
    # SLOWNESS VECTORS OF TRANSMITTED WAVES
    ########################################

    # Compute the coefficients of the bicubic equation from the stiffness
    # matrix of the upper layer, density, and the two horizontal components
    # of slowness.
    A, B, C, D = monoclinic_bicubic_coeffs(s[0], s[1], p2, C2)
    # Input the computed coefficients and solve the bicubic polynomial
    z = np.sort(np.roots(np.array([A, B, C, D])))
    s2P = np.array([s[0], s[1], np.sqrt(np.abs(z[0]))])
    s2S = np.array([s[0], s[1], np.sqrt(np.abs(z[1]))])
    s2T = np.array([s[0], s[1], np.sqrt(np.abs(z[2]))])

    ########################################
    # POLARIZATION VECTORS OF REFLECTED WAVES
    ########################################

    # Construct the Christoffel matrices for the three upper phases,
    # and compute the eigensystem of the matrices.

    # #### REFLECTED QUASI-P PHASE
    CM1P = christoffel(C1, s1P)
    w, v = np.linalg.eig(CM1P)
    ev1P = v[:, 0]

    # #### REFLECTED QUASI-S PHASE
    CM1S = christoffel(C1, s1S)
    w, v = np.linalg.eig(CM1S)
    ev1S = v[:, 1]

    # #### REFLECTED QUASI-T PHASE
    CM1T = christoffel(C1, s1T)
    w, v = np.linalg.eig(CM1T)
    ev1T = -v[:, 2]

    # Match up quasi-SV and quasi-SH with the proper eigenvalues/eigenvectors
    if np.sum(ev1T*n) > np.sum(ev1S*n):
        print('swapping upper shears')
        foo = s1S
        s1S = s1T
        s1T = foo

        foo = ev1S
        ev1S = ev1T
        ev1T = foo

    ########################################
    # POLARIZATION VECTORS OF TRANSMITTED WAVES
    ########################################

    # #### LOWER LAYER P PHASE
    CM2P = christoffel(C2, s2P)
    w, v = np.linalg.eig(CM2P)
    ev2P = v[:, 0]

    # #### LOWER LAYER S PHASE
    CM2S = christoffel(C2, s2S)
    w, v = np.linalg.eig(CM2S)
    ev2S = v[:, 1]

    # #### LOWER LAYER T PHASE
    CM2T = christoffel(C2, s2T)
    w, v = np.linalg.eig(CM2T)
    ev2T = -v[:, 2]

    # Match up quasi-SV and quasi-SH with the proper eigenvalues/eigenvectors
    if np.sum(ev2T*n) > np.sum(ev2S*n):
        print('swapping lower shears')
        foo = s2S
        s2S = s2T
        s2T = foo

        foo = ev2S
        ev2S = ev2T
        ev2T = foo

    # Construct X, Y, X', and Y' impedance matrices
    # #### UPPER MEDIUM
    X1 = np.zeros(shape=(3, 3))
    X1[0][0] = ev1P[0]
    X1[0][1] = ev1S[0]
    X1[0][2] = ev1T[0]
    X1[1][0] = ev1P[1]
    X1[1][1] = ev1S[1]
    X1[1][2] = ev1T[1]
    X1[2][0] = -(C1[0][2]*ev1P[0] + C1[2][5]*ev1P[1])*s1P[0] - (C1[1][2]*ev1P[1] + C1[2][5]*ev1P[0])*s1P[1] - C1[2][2]*ev1P[2]*s1P[2]
    X1[2][1] = -(C1[0][2]*ev1S[0] + C1[2][5]*ev1S[1])*s1S[0] - (C1[1][2]*ev1S[1] + C1[2][5]*ev1S[0])*s1S[1] - C1[2][2]*ev1S[2]*s1S[2]
    X1[2][2] = -(C1[0][2]*ev1T[0] + C1[2][5]*ev1S[1])*s1T[0] - (C1[1][2]*ev1T[1] + C1[2][5]*ev1T[0])*s1T[1] - C1[2][2]*ev1T[2]*s1T[2]

    Y1 = np.zeros(shape=(3, 3))
    Y1[0][0] = (-(C1[4][4]*s1P[0] + C1[3][4]*s1P[1])*ev1P[2] -
                 (C1[4][4]*ev1P[0] + C1[3][4]*ev1P[1])*s1P[2])
    Y1[0][1] = (-(C1[4][4]*s1S[0] + C1[3][4]*s1S[1])*ev1S[2] -
                 (C1[4][4]*ev1S[0] + C1[3][4]*ev1S[1])*s1S[2])
    Y1[0][2] = (-(C1[4][4]*s1T[0] + C1[3][4]*s1T[1])*ev1T[2] -
                 (C1[4][4]*ev1T[0] + C1[3][4]*ev1T[1])*s1T[2])
    Y1[1][0] = (-(C1[3][4]*s1P[0] + C1[3][3]*s1P[1])*ev1P[2] -
                 (C1[3][4]*ev1P[0] + C1[3][3]*ev1P[1])*s1P[2])
    Y1[1][1] = (-(C1[3][4]*s1S[0] + C1[3][3]*s1S[1])*ev1S[2] -
                 (C1[3][4]*ev1S[0] + C1[3][3]*ev1S[1])*s1S[2])
    Y1[1][2] = (-(C1[3][4]*s1T[0] + C1[3][3]*s1T[1])*ev1T[2] -
                 (C1[3][4]*ev1T[0] + C1[3][3]*ev1T[1])*s1T[2])
    Y1[2][0] = ev1P[2]
    Y1[2][1] = ev1S[2]
    Y1[2][2] = ev1T[2]

    # ### LOWER MEDIUM
    X2 = np.zeros(shape=(3, 3))
    X2[0][0] = ev2P[0]
    X2[0][1] = ev2S[0]
    X2[0][2] = ev2T[0]
    X2[1][0] = ev2P[1]
    X2[1][1] = ev2S[1]
    X2[1][2] = ev2T[1]
    X2[2][0] = -(C2[0][2]*ev2P[0] + C2[2][5]*ev2P[1])*s2P[0] - (C2[1][2]*ev2P[1] + C2[2][5]*ev2P[0])*s2P[1] - C2[2][2]*ev2P[2]*s2P[2]
    X2[2][1] = -(C2[0][2]*ev2S[0] + C2[2][5]*ev2S[1])*s2S[0] - (C2[1][2]*ev2S[1] + C2[2][5]*ev2S[0])*s2S[1] - C2[2][2]*ev2S[2]*s2S[2]
    X2[2][2] = -(C2[0][2]*ev2T[0] + C2[2][5]*ev2S[1])*s2T[0] - (C2[1][2]*ev2T[1] + C2[2][5]*ev2T[0])*s2T[1] - C2[2][2]*ev2T[2]*s2T[2]

    Y2 = np.zeros(shape=(3, 3))
    Y2[0][0] = (-(C2[4][4]*s2P[0] +
                  C2[3][4]*s2P[1])*ev2P[2] -
                 (C2[4][4]*ev2P[0] +
                  C2[3][4]*ev2P[1])*s2P[2])
    Y2[0][1] = (-(C2[4][4]*s2S[0] +
                  C2[3][4]*s2S[1])*ev2S[2] -
                 (C2[4][4]*ev2S[0] + C2[3][4]*ev2S[1])*s2S[2])
    Y2[0][2] = (-(C2[4][4]*s2T[0] + C2[3][4]*s2T[1])*ev2T[2] -
                 (C2[4][4]*ev2T[0] + C2[3][4]*ev2T[1])*s2T[2])
    Y2[1][0] = (-(C2[3][4]*s2P[0] + C2[3][3]*s2P[1])*ev2P[2] -
                 (C2[3][4]*ev2P[0] + C2[3][3]*ev2P[1])*s2P[2])
    Y2[1][1] = (-(C2[3][4]*s2S[0] + C2[3][3]*s2S[1])*ev2S[2] -
                 (C2[3][4]*ev2S[0] + C2[3][3]*ev2S[1])*s2S[2])
    Y2[1][2] = (-(C2[3][4]*s2T[0] + C2[3][3]*s2T[1])*ev2T[2] -
                 (C2[3][4]*ev2T[0] + C2[3][3]*ev2T[1])*s2T[2])
    Y2[2][0] = ev2P[2]
    Y2[2][1] = ev2S[2]
    Y2[2][2] = ev2T[2]

    # Solve the X, Y, X', Y' system of equations for
    # the Zoeppritz reflection matrix R
    T = np.linalg.inv(np.linalg.inv(X1).dot(X2) + np.linalg.inv(Y1).dot(Y2))
    R = (np.linalg.inv(X1).dot(X2) - np.linalg.inv(Y1).dot(Y2)).dot(T)

    return(R[0][0])


def monoclinic_bicubic_coeffs(s1, s2, p, C):
    c11 = C[0][0]
    c22 = C[1][1]
    c33 = C[2][2]
    c44 = C[3][3]
    c55 = C[4][4]
    c66 = C[5][5]
    c12 = C[0][1]
    c13 = C[0][2]
    c23 = C[1][2]
    c16 = C[0][5]
    c26 = C[1][5]
    c36 = C[2][5]
    c45 = C[3][4]
    A = (c33*c44*c55 - c33*c45**2)
    B = (c11*c33*c44*s1**2 - 2*c12*c33*c45*s1*s2 - c13**2*c44*s1**2 +
         2*c13*c23*c45*s1*s2 - 2*c13*c36*c44*s1*s2 + 2*c13*c36*c45*s1**2 -
         2*c13*c44*c55*s1**2 + 2*c13*c45**2*s1**2 + 2*c16*c33*c44*s1*s2 -
         2*c16*c33*c45*s1**2 + c22*c33*c55*s2**2 - c23**2*c55*s2**2 +
         2*c23*c36*c45*s2**2 - 2*c23*c36*c55*s1*s2 - 2*c23*c44*c55*s2**2 +
         2*c23*c45**2*s2**2 - 2*c26*c33*c45*s2**2 + 2*c26*c33*c55*s1*s2 +
         c33*c44*c66*s2**2 - c33*c44*p - 2*c33*c45*c66*s1*s2 +
         c33*c55*c66*s1**2 - c33*c55*p - c36**2*c44*s2**2 +
         2*c36**2*c45*s1*s2 - c36**2*c55*s1**2 - 4*c36*c44*c55*s1*s2 +
         4*c36*c45**2*s1*s2 - c44*c55*p + c45**2*p)
    C = (c11*c22*c33*s1**2*s2**2 - c11*c23**2*s1**2*s2**2 -
         2*c11*c23*c36*s1**3*s2 - 2*c11*c23*c44*s1**2*s2**2 -
         2*c11*c23*c45*s1**3*s2 + 2*c11*c26*c33*s1**3*s2 +
         c11*c33*c66*s1**4 - c11*c33*p*s1**2 - c11*c36**2*s1**4 -
         2*c11*c36*c44*s1**3*s2 - 2*c11*c36*c45*s1**4 + c11*c44*c55*s1**4 -
         c11*c44*p*s1**2 - c11*c45**2*s1**4 - c12**2*c33*s1**2*s2**2 +
         2*c12*c13*c23*s1**2*s2**2 + 2*c12*c13*c36*s1**3*s2 +
         2*c12*c13*c44*s1**2*s2**2 + 2*c12*c13*c45*s1**3*s2 -
         2*c12*c16*c33*s1**3*s2 + 2*c12*c23*c36*s1*s2**3 +
         2*c12*c23*c45*s1*s2**3 + 2*c12*c23*c55*s1**2*s2**2 -
         2*c12*c26*c33*s1*s2**3 - 2*c12*c33*c66*s1**2*s2**2 +
         2*c12*c36**2*s1**2*s2**2 + 2*c12*c36*c44*s1*s2**3 +
         4*c12*c36*c45*s1**2*s2**2 + 2*c12*c36*c55*s1**3*s2 +
         2*c12*c44*c55*s1**2*s2**2 - 2*c12*c45**2*s1**2*s2**2 +
         2*c12*c45*p*s1*s2 - c13**2*c22*s1**2*s2**2 -
         2*c13**2*c26*s1**3*s2 - c13**2*c66*s1**4 + c13**2*p*s1**2 +
         2*c13*c16*c23*s1**3*s2 + 2*c13*c16*c36*s1**4 +
         2*c13*c16*c44*s1**3*s2 + 2*c13*c16*c45*s1**4 -
         2*c13*c22*c36*s1*s2**3 - 2*c13*c22*c45*s1*s2**3 -
         2*c13*c22*c55*s1**2*s2**2 + 2*c13*c23*c26*s1*s2**3 +
         2*c13*c23*c66*s1**2*s2**2 - 2*c13*c26*c36*s1**2*s2**2 +
         2*c13*c26*c44*s1*s2**3 - 2*c13*c26*c45*s1**2*s2**2 -
         4*c13*c26*c55*s1**3*s2 + 2*c13*c36*p*s1*s2 +
         2*c13*c44*c66*s1**2*s2**2 + 2*c13*c45*p*s1*s2 - 2*c13*c55*c66*s1**4 +
         2*c13*c55*p*s1**2 - c16**2*c33*s1**4 + 2*c16*c22*c33*s1*s2**3 -
         2*c16*c23**2*s1*s2**3 - 2*c16*c23*c36*s1**2*s2**2 -
         4*c16*c23*c44*s1*s2**3 - 2*c16*c23*c45*s1**2*s2**2 +
         2*c16*c23*c55*s1**3*s2 + 2*c16*c26*c33*s1**2*s2**2 -
         2*c16*c33*p*s1*s2 - 2*c16*c36*c44*s1**2*s2**2 + 2*c16*c36*c55*s1**4 +
         4*c16*c44*c55*s1**3*s2 - 2*c16*c44*p*s1*s2 - 4*c16*c45**2*s1**3*s2 +
         2*c16*c45*p*s1**2 + c22*c33*c66*s2**4 - c22*c33*p*s2**2 -
         c22*c36**2*s2**4 - 2*c22*c36*c45*s2**4 - 2*c22*c36*c55*s1*s2**3 +
         c22*c44*c55*s2**4 - c22*c45**2*s2**4 - c22*c55*p*s2**2 -
         c23**2*c66*s2**4 + c23**2*p*s2**2 + 2*c23*c26*c36*s2**4 +
         2*c23*c26*c45*s2**4 + 2*c23*c26*c55*s1*s2**3 + 2*c23*c36*p*s1*s2 -
         2*c23*c44*c66*s2**4 + 2*c23*c44*p*s2**2 + 2*c23*c45*p*s1*s2 +
         2*c23*c55*c66*s1**2*s2**2 - c26**2*c33*s2**4 - 2*c26*c33*p*s1*s2 +
         2*c26*c36*c44*s2**4 - 2*c26*c36*c55*s1**2*s2**2 +
         4*c26*c44*c55*s1*s2**3 - 4*c26*c45**2*s1*s2**3 +
         2*c26*c45*p*s2**2 - 2*c26*c55*p*s1*s2 - c33*c66*p*s1**2 -
         c33*c66*p*s2**2 + c33*p**2 + c36**2*p*s1**2 +
         c36**2*p*s2**2 + 2*c36*c44*p*s1*s2 + 2*c36*c45*p*s1**2 +
         2*c36*c45*p*s2**2 + 2*c36*c55*p*s1*s2 + 4*c44*c55*c66*s1**2*s2**2 -
         c44*c55*p*s1**2 - c44*c55*p*s2**2 - c44*c66*p*s2**2 +
         c44*p**2 - 4*c45**2*c66*s1**2*s2**2 + c45**2*p*s1**2 +
         c45**2*p*s2**2 + 2*c45*c66*p*s1*s2 - c55*c66*p*s1**2 + c55*p**2)
    D = (c11*c22*c44*s1**2*s2**4 + 2*c11*c22*c45*s1**3*s2**3 +
         c11*c22*c55*s1**4*s2**2 - c11*c22*p*s1**2*s2**2 +
         2*c11*c26*c44*s1**3*s2**3 + 4*c11*c26*c45*s1**4*s2**2 +
         2*c11*c26*c55*s1**5*s2 - 2*c11*c26*p*s1**3*s2 +
         c11*c44*c66*s1**4*s2**2 - c11*c44*p*s1**2*s2**2 +
         2*c11*c45*c66*s1**5*s2 - 2*c11*c45*p*s1**3*s2 +
         c11*c55*c66*s1**6 - c11*c55*p*s1**4 - c11*c66*p*s1**4 +
         c11*p**2*s1**2 - c12**2*c44*s1**2*s2**4 - 2*c12**2*c45*s1**3*s2**3 -
         c12**2*c55*s1**4*s2**2 + c12**2*p*s1**2*s2**2 -
         2*c12*c16*c44*s1**3*s2**3 - 4*c12*c16*c45*s1**4*s2**2 -
         2*c12*c16*c55*s1**5*s2 + 2*c12*c16*p*s1**3*s2 -
         2*c12*c26*c44*s1*s2**5 - 4*c12*c26*c45*s1**2*s2**4 -
         2*c12*c26*c55*s1**3*s2**3 + 2*c12*c26*p*s1*s2**3 -
         2*c12*c44*c66*s1**2*s2**4 - 4*c12*c45*c66*s1**3*s2**3 -
         2*c12*c55*c66*s1**4*s2**2 + 2*c12*c66*p*s1**2*s2**2 -
         c16**2*c44*s1**4*s2**2 - 2*c16**2*c45*s1**5*s2 - c16**2*c55*s1**6 +
         c16**2*p*s1**4 + 2*c16*c22*c44*s1*s2**5 + 4*c16*c22*c45*s1**2*s2**4 +
         2*c16*c22*c55*s1**3*s2**3 - 2*c16*c22*p*s1*s2**3 +
         2*c16*c26*c44*s1**2*s2**4 + 4*c16*c26*c45*s1**3*s2**3 +
         2*c16*c26*c55*s1**4*s2**2 - 2*c16*c26*p*s1**2*s2**2 -
         2*c16*c44*p*s1*s2**3 - 4*c16*c45*p*s1**2*s2**2 -
         2*c16*c55*p*s1**3*s2 + 2*c16*p**2*s1*s2 + c22*c44*c66*s2**6 -
         c22*c44*p*s2**4 + 2*c22*c45*c66*s1*s2**5 - 2*c22*c45*p*s1*s2**3 +
         c22*c55*c66*s1**2*s2**4 - c22*c55*p*s1**2*s2**2 -
         c22*c66*p*s2**4 + c22*p**2*s2**2 - c26**2*c44*s2**6 -
         2*c26**2*c45*s1*s2**5 - c26**2*c55*s1**2*s2**4 + c26**2*p*s2**4 -
         2*c26*c44*p*s1*s2**3 - 4*c26*c45*p*s1**2*s2**2 -
         2*c26*c55*p*s1**3*s2 + 2*c26*p**2*s1*s2 - c44*c66*p*s1**2*s2**2 -
         c44*c66*p*s2**4 + c44*p**2*s2**2 - 2*c45*c66*p*s1**3*s2 -
         2*c45*c66*p*s1*s2**3 + 2*c45*p**2*s1*s2 - c55*c66*p*s1**4 -
         c55*c66*p*s1**2*s2**2 + c55*p**2*s1**2 + c66*p**2*s1**2 +
         c66*p**2*s2**2 - p**3)
    return(A, B, C, D)


def christoffel(C, s):
    CM = np.zeros(shape=(3, 3))
    CM[0][0] = (C[0][0]*s[0]**2 +
                C[5][5]*s[1]**2 +
                C[4][4]*s[2]**2 +
                2*C[0][5]*s[0]*s[1])
    CM[0][1] = (C[0][5]*s[0]**2 +
                C[1][5]*s[1]**2 +
                C[3][4]*s[2]**2 +
                (C[0][1]+C[5][5])*s[0]*s[1])
    CM[0][2] = ((C[0][2]+C[4][4])*s[0]*s[2] +
                (C[2][5]+C[3][4])*s[1]*s[2])
    CM[1][0] = (C[0][5]*s[0]**2 +
                C[1][5]*s[1]**2 +
                C[3][4]*s[2]**2 +
                (C[0][1]+C[5][5])*s[0]*s[1])
    CM[1][1] = (C[5][5]*s[0]**2 +
                C[1][1]*s[1]**2 +
                C[3][3]*s[2]**2 +
                2*C[1][5]*s[0]*s[1])
    CM[1][2] = ((C[2][5]+C[3][4])*s[0]*s[2] +
                (C[1][2]+C[3][3])*s[1]*s[2])
    CM[2][0] = ((C[0][2]+C[4][4])*s[0]*s[2] +
                (C[2][5]+C[3][4])*s[1]*s[2])
    CM[2][1] = ((C[2][5]+C[3][4])*s[0]*s[2] +
                (C[3][3]+C[1][2])*s[1]*s[2])
    CM[2][2] = (C[4][4]*s[0]**2 +
                C[3][3]*s[1]**2+C[2][2]*s[2]**2 +
                2*C[3][4]*s[0]*s[1])

    return(CM)


def vavrycuk_psencik_hti(vp1, vs1, p1, d1, e1, y1,
                         vp2, vs2, p2, d2, e2, y2,
                         phi, theta1):
    """
    Reflectivity for arbitrarily oriented HTI media, using the formulation
    derived by Vavrycuk and Psencik [1998], "PP-wave reflection coefficients
    in weakly anisotropic elastic media"
    """
    theta1 = np.radians(theta1)
    phi = np.radians(phi)

    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)
    theta = (theta1 + theta2)/2
    theta = theta1
    G1 = p1*(vs1**2)
    G2 = p2*(vs2**2)
    Z1 = p1*vp1
    Z2 = p2*vp2

    a = (vp1 + vp2)/2
    B = (vs1 + vs2)/2
    Z = (Z1 + Z2)/2
    G = (G1 + G2)/2

    dZ = Z2 - Z1
    da = vp2 - vp1
    dG = G2 - G1
    dd = d2 - d1
    de = e2 - e1
    dy = y2 - y1

    A = (1/2*(dZ/Z) +
         1/2*(da/a)*np.tan(theta)**2 -
         2*((B/a)**2)*(dG/G)*np.sin(theta)**2)
    B = 1/2*(dd*(np.cos(phi)**2) - 8*((B/a)**2)*dy*(np.sin(phi)*2))
    C = 1/2*(de*(np.cos(phi)**4) + dd*(np.cos(phi)**2)*(np.sin(phi)**2))

    Rpp = A + B*np.sin(theta)**2 + C*np.sin(theta)**2*np.tan(theta)**2

    return(Rpp)
