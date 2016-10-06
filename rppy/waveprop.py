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


def synthetic_aki_richards(V, p, d):
    """
    Compute synthetic for a layered medium using the propagator matrix
    method of Aki and Richards (1980) and Claerbout (1985).

    The synthetic seismogram is computed as a product of propagator matrices,
    one for each layer in the stack.

    DOESN'T ACTUALLY WORK YET. NEEDS COMPLETION
    """
    for w in freqs:
        S1 =
        W1 =
        for n in xrange(1, numlayer):
            S = S * S1 * A(n, w)
            W = W * W1 * A(n, w)

    s = np.fft.ifft(S)
    w = np.fft.ifft(W)


def A(n, w, d, V, p):
    """
    Compute the "layer matrix" of Aki and Richards (1980) and Claerbout (1985).
    Internal function for use in 'synthetic_aki_richards'
    """
    wdV = w*d/V
    A = np.zeros(shape=(2, 2))
    A = [[np.cos(wdV), j*p*V*np.sin(wdV)],
         [(j / (p*V))*np.sin(wdV), np.cos(wdV)]]


def synthetic_kennett():
    """
    Compute synthetic for a layered medium using the invariant imbedding method
    of Kennett (1974, 1983).
    """

    M = np.zeros(shape=(4, 4))
    N = np.zeros(shape=(4, 4))

    M[0][0] = -np.sin(theta[k-1])
    M[0][1] = -np.cos(phi[k-1])
    M[0][2] = np.sin(theta[k])
    M[0][3] = np.cos(phi[k])],
    M[1][0] = np.cos(theta[k-1])
    M[1][1] = -np.sin(phi[k-1])
    M[1][2] = np.cos(theta[k])
    M[1][3] = -np.sin(phi[k])
    M[2][0] = 2*Is[k-1]*np.sin(phi[k-1])*np.cos(theta[k-1])
    M[2][1] = Is[k-1](1 - 2*np.sin(phi[k-1])^2)
    M[2][2] = 2*Is[k]*np.sin(phi[k])*np.cos(theta[k])
    M[2][3] = Is[k]*(1 - 2*np.sin(phi[k])^2)
    M[3][0] = -Ip[k-1]*(1 - 2*np.sin(phi[k-1])^2)
    M[3][1] = Is[k-1]*np.sin(2*phi[k-1])
    M[3][2] = Ip[k]*(1 - 2*np.sin(phi[k])^2)
    M[3][3] = -Is[k]*np.sin(2*phi[k])]]

    N[0][0] = -np.sin(theta[k-1])
    N[0][1] = np.cos(phi[k-1])
    N[0][2] = -np.sin(theta[k])
    N[0][3] = -np.cos(phi[k])
    N[1][0] = -np.cos(theta[k-1])
    N[1][1] = -np.sin(phi[k-1])
    N[1][2] = np.cos(theta[k])
    N[1][3] = -np.sin(phi[k])
    N[2][0] = 2*Is[k-1]*np.sin(phi[k-1])*np.cos(theta[k-1])
    N[2][1] = Is[k-1]*(1 - 2*np.sin(phi[k-1])^2)
    N[2][2] = 2*Is[k]*np.sin(phi[k])*np.cos(theta[k])
    N[2][3] = Is[k]*(1 - 2*np.sin(phi[k])^2)
    N[3][0] = Ip[k-1]*(1 - 2*np.sin(phi[k-1])^2)
    N[3][1] = -Is[k-1]*np.sin(2*phi[k-1])
    N[3][2] = -Ip[k]*(1 - 2*np.sin(phi[k])^2)
    N[3][3] = Is[k]*np.sin(2*phi[k])

    iMN = np.linalg.inv(M)*N


def RDhat(k):
    RDhat = RD(k) + TU(k)*ED(k)*RDhat(k+1)*ED(k)*np.invert(I - RU(k)*ED(k)*RDhat(k+1)*ED(k))*TD(k)


def RD(k):
    RD = np.zeros(shape=(2, 2))
    RD[0][0] = iMN[0][0]
    RD[0][1] = iMN[0][1]*np.sqrt(((Vp[k-1]*np.cos(theta[k-1]))/(Vs[k-1]*np.cos(phi[k-1]))))
    RD[1][0] = iMN[1][0]*np.sqrt((Vs[k-1]*np.cos(phi[k-1]))/(Vp[k-1]*np.cos(theta[k-1])))
    RD[1][1] = iMN[1][1]


def TD(k):
    TD = np.zeros(shape=(2, 2))
    TD[0][0] = iMN[2][0]*np.sqrt()
    TD[0][1] = iMN[2][1]*np.sqrt()
    TD[1][0] = iMN[3][0]*np.sqrt()
    TD[1][1] = iMN[3][1]*np.sqrt()


def RU(k):
    RU = np.zeros(shape=(2, 2))
    RU[0][0] = PPud
    RU[0][1] = SPud*np.sqrt()
    RU[1][0] = PSud*np.sqrt()
    RU[1][1] = SSud


def TU(k):
    TU = np.zeros(shape=(2, 2))
    TU[0][0] = PPuu*np.sqrt()
    TU[0][1] = SPuu*np.sqrt()
    TU[1][0] = PSuu*np.sqrt()
    TU[1][1] = SSuu*np.sqrt()
