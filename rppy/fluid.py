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


def batzle_wang(P, T, fluid, S=None, G=None, api=None):
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
        Rg = 2.03*G*(P*np.exp(0.02878*api - 0.00377*T))**1.205
        rho0 = 141.5 / (api + 131.5)
        B0 = 0.972 + 0.00038*(2.4*Rg*(G/rho0)**0.5 + T + 17.8)**(1.175)

        rho_r = (rho0/B0)*(1 + 0.001*Rg)**-1    # pseudo-density of oil
        rhog = (rho0 + 0.0012*G*Rg)/B0          # density of oil with gas
        rhop = (rhog + (0.00277*P -             # correct for pressure
                1.71e-7*P**3)*(rhog - 1.15)**2 + 3.49e-4*P)

        rho = rhop / (0.972 + 3.81e-4*(T + 17.78)**1.175)  # correct for temp
        Vp = 2096*(rho_r / (2.6 - rho_r))**0.5 - 3.7*T + 4.64*P + 0.0115*(
            4.12*(1.08/rho_r - 1)**0.5 -1)*T*P

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

