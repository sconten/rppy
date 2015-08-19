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


def han_eberhart_phillips(phi, C, Pe):
    """
    Calculate the Eberhart-Phillips multivariate version of Han's velocity -
    porosity relation for water-saturated shaley sandstones.

    :param phi: Porosity [frac.]
    :param C: Vshale [frac.]
    :param Pe: Effective pressure [Mpa]
    """

    Vp = (5.77 - 6.94*phi - 1.73*np.sqrt(C) +
          0.446*(0.01*Pe - np.exp(-16.7*0.01*Pe)))
    Vs = (3.70 - 4.94*phi - 1.57*np.sqrt(C) +
          0.361*(0.01*Pe - np.exp(-16.7*0.01*Pe)))

    return(Vp, Vs)


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


def hudson():
    """
    Hudson's model for cracked media, based on a scattering-theory analysis of
    the mean wavefield in an elastic solid with thin, p[enny-shaped ellipsoidal
    cracks or inclusions (Hudson, 1980)
    """

    q = 15*(l**2)/(u**2) + 28*(l/u) + 28

    U1 = (16*(l + 2*u))/(3*(3*l + 4*u))
    U3 = (4*(l + 2*u))/(3*(l + u))

    C1[0][0] = -l**2/u*e*U3
    C1[0][2] = l(l + 2*u) / u*e*U3
    C1[2][2] = (l + 2*u)**2 / u*e*U3
    C1[3][3] = -u*e*U1
    C1[5][5] = 0

    C2[0][0] = q/15*l**2/(l+2*u)*(e*U3)**2
    C2[0][2] = q/15*l*(e*U3)**2
    C2[2][2] = q/15*(l+2*u)*(e*U3)**2
    C2[3][3] = 2/15*u*(3*l+8*u)/(l+2*u)*(e*U1)**2
    C2[5][5] = 0
    Ceff = C0 + C1 + C2

def han(phi, C):
    """
    Han [1986] model for water-saturated sandstones at 40 MPa
    """

    Vp = 5.59 - 6.93*phi - 2.13*C
    Vs = 3.52 - 4.91*phi - 1.89*C

    return(Vp, Vs)


def hertz_mindlin(u, v, P, phi, n=None):
    """
    Elastic moduli of an elastic sphere pack subject to confing pressure given
    by the Hertz-Mindlin [Mindlin, 1949] theory.

    If the coordination number n is not given, the function uses the empirical
    dependency of n on porosity shown by Murphy [1982]
    """

    if not n:
        n = 20 - 34*phi + 14*phi**2

    Khm = ((n**2*(1-phi)**2*u**2*P) / (18*np.pi**2*(1-v)**2))**(1/3)
    uhm = ((5-4*v)/(10-5*v)) * ((3*n**2*(1-phi)**2*u**2*P) /
                                (2*np.pi**2*(1-v)**2))**(1/3)

    return(Khm, uhm)
