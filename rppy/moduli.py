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
    if v is not None and u is not None:
        E = 2*u*(1+v)
    elif v is not None and K is not None:
        E = 3*K*(1-2*v)
    elif v is not None and L is not None:
        E = (L*(1+v)*(1-2*v))/(v)
    elif u is not None and K is not None:
        E = (9*K*u)/(3*K+u)
    elif u is not None and L is not None:
        E = u*(3*L+2*u)/(L+u)
    elif K is not None and L is not None:
        E = 9*K*(K-L)/(3*K-L)
    elif Vp is not None and Vs is not None and rho is not None:
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
    if E is not None and u is not None:
        v = (E-2*u)/(2*u)
    elif E is not None and K is not None:
        v = (3*K - E)/(6*K)
    elif E is not None and L is not None:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        v = (2*L)/(E+L+R)
    elif u is not None and K is not None:
        v = (3*K-2*u)/(6*K + 2*u)
    elif u is not None and L is not None:
        v = L/(2*(L+u))
    elif K is not None and L is not None:
        v = L/(3*K-L)
    elif Vp is not None and Vs is not None and rho is not None:
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
    if E is not None and v is not None:
        u = E/(2*(1+v))
    elif E is not None and K is not None:
        u = 3*K*E/(9*K-E)
    elif E is not None and L is not None:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        u = (E-3*L+R)/4
    elif v is not None and K is not None:
        u = 3*K*(1-2*v)/(2*(1+v))
    elif v is not None and L is not None:
        u = L*(1-2*v)/(2*v)
    elif K is not None and L is not None:
        u = (3/2)*(K-L)
    elif Vp is not None and Vs is not None and rho is not None:
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
    if E is not None and v is not None:
        K = E/(3*(1-2*v))
    elif E is not None and u is not None:
        K = E*u/(3*(3*u-E))
    elif E is not None and L is not None:
        R = np.sqrt(E**2 + 9*L**2 + 2*E*L)
        K = (E+3*L+R)/6
    elif v is not None and u is not None:
        K = 2*u*(1+v)/(3*(1-2*v))
    elif v is not None and L is not None:
        K = L*(1+v)/(3*v)
    elif u is not None and L is not None:
        K = (3*L+2*u)/3
    elif Vp is not None and Vs is not None and rho is not None:
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
    if E is not None and v is not None:
        L = E*v/((1+v)*(1-2*v))
    elif E is not None and u is not None:
        L = u*(E - 2*u)/(3*u - E)
    elif E is not None and K is not None:
        L = 3*K*(3*K-E)/(9*K-E)
    elif v is not None and u is not None:
        L = 2*u*v/(1-2*v)
    elif v is not None and K is not None:
        L = 3*K*v/(1+v)
    elif u is not None and K is not None:
        L = (3*K-2*u)/3
    elif Vp is not None and Vs is not None and rho is not None:
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
    if E is not None and v is not None:
        u = shear(E=E, v=v)
        K = bulk(E=E, v=v)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E is not None and u is not None:
        K = bulk(E=E, u=u)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E is not None and K is not None:
        u = shear(E=E, K=K)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif E is not None and L is not None:
        K = bulk(E=E, L=L)
        u = shear(E=E, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v is not None and u is not None:
        K = bulk(v=v, u=u)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v is not None and K is not None:
        u = shear(v=v, K=K)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif v is not None and L is not None:
        K = bulk(v=v, L=L)
        u = shear(v=v, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif u is not None and K is not None:
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif u is not None and L is not None:
        K = bulk(u=u, L=L)
        Vp = np.sqrt((K + 4/3*u)/rho)
    elif K is not None and L is not None:
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
    if u is not None:
        Vs = np.sqrt(u/rho)
    elif E is not None and v is not None:
        u = shear(E=E, v=v)
        Vs = np.sqrt(u/rho)
    elif E is not None and K is not None:
        u = shear(E=E, K=K)
        Vs = np.sqrt(u/rho)
    elif E is not None and L is not None:
        u = shear(E=E, L=L)
        Vs = np.sqrt(u/rho)
    elif v is not None and K is not None:
        u = shear(v=v, K=K)
        Vs = np.sqrt(u/rho)
    elif v is not None and L is not None:
        u = shear(v=v, L=L)
        Vs = np.sqrt(u/rho)
    elif K is not None and L is not None:
        u = shear(K=K, L=L)
        Vs = np.sqrt(u/rho)
    else:
        Vs = None
    return(Vs)
