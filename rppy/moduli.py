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
