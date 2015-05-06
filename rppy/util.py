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
