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


def Rd0(rho1, V1, rho2, V2):
    """
    Normal incidence, single-interface downward-propagating reflection matrix
    """
    Rd0 = (rho1*V1 - rho2*V2) / (rho1*V1 + rho2*V2)

    return(Rd0)


def Tu0(rho1, V1, rho2, V2):
    """
    Normal incidence, single-interface upward-propagating transmission matrix
    """
    Tu0 = (2*np.sqrt(rho1*V1*rho2*V2)) / (rho1*V1 + rho2*V2)

    return(Tu0)


def Ed0(w, dk, Vp2, Vs2, theta2, phi2):
    """
    Normal incidence, singler-interface phase shift operator
    """
    Ed0 = np.zeros(shape=(2, 2))
    Ed0[0][0] = np.exp(i*w*dk*np.cos(theta2)/Vp2)
    Ed0[0][1] = 0
    Ed0[1][0] = 0
    Ed0[1][1] = np.exp(i*w*dk*np.cos(phi2)/Vs2)

    return(Ed0)


def Rd(PP_du, SP_du, PS_du, SS_du, Vp1, Vs1, theta1, phi1):
    """
    Single interface donward-propagating reflection matrix
    """
    Rd = np.zeros(shape=(2, 2))
    Rd[0][0] = PP_du
    Rd[0][1] = SP_du*np.sqrt((Vp1*np.cos(theta1) / Vs1*np.cos(phi1)))
    Rd[1][0] = PS_du*np.sqrt((Vs1*np.cos(phi1) / Vp1*np.cos(theta1)))
    Rd[1][1] = SS_du

    return(Rd)


def Td(PP_dd, SP_dd, PS_dd, SS_dd, rho1, Vp1, Vs1, rho2, Vp2, Vs2, theta1, phi1, theta2, phi2):
    """
    Single interface downward-propagating transmission matrix
    """
    Td = np.zeros(shape=(2, 2))
    Td[0][0] = PP_dd*np.sqrt((rho2*Vp2*np.cos(theta2)) / (rho1*Vp1*np.cos(theta1)))
    Td[0][1] = SP_dd*np.sqrt((rho2*Vp2*np.cos(theta2)) / (rho1*Vs1*np.cos(phi1)))
    Td[1][0] = PS_dd*np.sqrt((rho2*Vs2*np.cos(phi2)) / (rho1*Vp1*np.cos(theta1)))
    Td[1][1] = SS_dd*np.sqrt((rho2*Vs2*np.cos(phi2)) / (rho1*Vs1*np.cos(phi1)))

    return(Td)


def Ru(PP_ud, SP_ud, PS_ud, SS_ud, Vp2, Vs2, theta2, phi2):
    """
    Single interface upward-propagating reflection matrix
    """
    Ru = np.zeros(shape=(2, 2))
    Ru[0][0] = PP_ud
    Ru[0][1] = SP_ud*np.sqrt((Vp2*np.cos(theta2)) / (Vs2*np.cos(phi2)))
    Ru[1][0] = PS_ud*np.sqrt((Vs2*np.cos(phi2)) / (Vp2*np.cos(theta2)))
    Ru[1][1] = SS_ud

    return(Ru)


def Tu(PP_uu, SP_uu, PS_uu, SS_uu, rho1, Vp1, Vs1, rho2, Vp2, Vs2, theta1, phi1, theta2, phi2):
    """
    Single interface downward-propagating transmission matrix
    """
    Tu = np.zeros(shape=(2, 2))
    Tu[0][0] = PP_uu*np.sqrt((rho1*Vp1*np.cos(theta1)) / (rho2*Vp2*np.cos(theta2)))
    Tu[0][1] = SP_uu*np.sqrt((rho1*Vp1*np.cos(theta1)) / (rho2*Vs2*np.cos(phi2)))
    Tu[1][0] = PS_uu*np.sqrt((rho1*Vs1*np.cos(phi1)) / (rho2*Vp2*np.cos(theta2)))
    Tu[1][1] = SS_uu*np.sqrt((rho1*Vs1*np.cos(phi1)) / (rho2*Vs2*np.cos(phi2)))

    return(Tu)


def full_waveform_synthetic():
    """
    Compute synthetic for a layerted medium using the propagator matrix
    method of Aki and Richard (1980) and Claerbout (1985).
    
    The synthetic seismogram is computed as a product of propagator matrices,
    one for each layer in the stack. The calculation is done in the frequency
    domain, and includes the effect of multiples.
    ""