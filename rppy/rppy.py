#    rppy - a geophysical library for Python
#    Copyright (C) 2015  Sean Matthew Contenti
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import matplotlib.pyplot as plt


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
    p = np.sin(theta1)/vp1      # Ray parameter
    if theta1 >= theta_crit_1:
        theta2 = None
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = np.arcsin(p*vs2)  # S-wave refraction
    elif theta1 >= theta_crit_2:
        theta2 = None
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = None  # S-wave refraction
    else:
        theta2 = np.arcsin(p*vp2)   # P-wave refraction
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = np.arcsin(p*vs2)  # S-wave refraction

        return(theta2, thetas1, thetas2, p)


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    Calculate the AVO response for a PP reflection based on the Shuey
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
    drho = rho2 - rho1
    dvs = vs2 - vs1
    rho = (rho1 + rho2) / 2.
    vs = (vs1 + vs2) / 2.
    vp = (vp1 + vp2) / 2.
    theta = (theta1 + theta2)/2

    R0 = 0.5*(dvp/vp + drho/rho)
    G = 0.5*dvp/vp - 2*vs**2/vp**2*(drho/rho + 2*dvs/vs)
    F = 0.5*dvp/vp

    Rpp = R0 + G*np.sin(theta)**2 + F*(np.tan(theta)**2 - np.sin(theta)**2)

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
    thetas = (thetas1 + thetas2) / 2.
    rho = (rho1 + rho2) / 2.
    vp = (vp1 + vp2) / 2.
    vs = (vs1 + vs2) / 2.

    Rpp = (0.5)*(1 - 4*p**2*vs**2)*(drho/rho) + (dvp/(2*np.cos(theta)**2*vp)) - (4*p**2*vs**2*dvs/vs)

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
    # Need reflection and refraction angled for Zoeppritz
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

    Rpp = ((1./2.)*np.log((vp2*rho2*np.cos(theta1)) /
           (vp1*rho1*np.cos(theta2))) + (np.sin(theta1)/vp1)**2*(vs1**2 -
           vs2**2)*(2. + np.log(rho2/rho1)/np.log(vs2/vs1)))

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
        L = np.average((1/(K + (4/3)*z)), weights=f)**-1 - (4/3)*z
        return (L)

    def HSgamma(z):
        G = np.average((1/(u + z)), weights=f)**(-1) - z
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


def youngs(E, v, u, K, L, Vp, Vs, rho):
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
        E = rho*Vs**2*(3*Vp**2 - 4*Vs**2)/(Vp**2 - Vs**2)
    else:
        E = None
    return(E)


def poissons(E, v, u, K, L, Vp, Vs, rho):
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


def shear(E, v, u, K, L, Vp, Vs, rho):
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


def bulk(E, v, u, K, L, Vp, Vs, rho):
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


def lame(E, v, u, K, L, Vp, Vs, rho):
    """
    Compute the first Lame's parameter of a material given sufficient other moduli.

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

    plt.plot(fw, v, 'r')
    plt.plot(fw, r, 'r')
    plt.plot(fw, hsu, 'r')
    plt.plot(fw, hsl, 'b')

    plt.axis([0, 1, 0, 35])
    plt.show()

    thetas = np.arange(1, 47, 1)
    Rppz = np.empty(np.shape(thetas))
    Rppb = np.empty(np.shape(thetas))
    Rppak = np.empty(np.shape(thetas))
    Rpps = np.empty(np.shape(thetas))

    plt.figure(2)
    for n in range(np.size(thetas)):
        dummy = zoeppritz(3000, 1500, 2000, 4000, 2000, 2200, np.radians(thetas[n]))
        Rppz[n] = dummy[0]
        Rppb[n] = bortfeld(3000, 1500, 2000, 4000, 2000, 2200, np.radians(thetas[n]))
        Rppak[n] = aki_richards(3000, 1500, 2000, 4000, 2000, 2200, np.radians(thetas[n]))
        Rpps[n] = shuey(3000, 1500, 2000, 4000, 2000, 2200, np.radians(thetas[n]))

    plt.plot(thetas, Rppz, thetas, Rppb, thetas, Rppak, thetas, Rpps)
    plt.legend(['Zoeppritz', 'Bortfeld', 'Aki-Richards', 'Shuey'])
    plt.xlim([20, 40])
    plt.ylim([0.14, 0.22])
    plt.show()


if __name__ == "__main__":
    main()
