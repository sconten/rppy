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
    dvp = vp2 - vp1
    drho = rho2 - rho1
    rho = (rho1 + rho2) / 2.
    vs = (vs1 + vs2) / 2.
    vp = (vp1 + vp2) / 2.
    dvs = vs2 - vs1
    Rpz = (1. / 2.)*((dvp / vp) + (drho / rho))
    B = (dvp/(2*vp) - (2*vs**2/vp**2)*(2*dvs/vs + drho/rho))
    C = dvp/(2*vp)

    Rpp = Rpz + B*np.sin(theta1)**2 + C*(np.tan(theta1)**2 - np.sin(theta1)**2)

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

    Rpp = (0.5*(1.-4.*(p**2)*(vs**2))*drho/rho +
           (1/(2.*np.cos(theta)))*(dvp/vp) - 4.*(p**2)*(vs**2)*dvs/vs)

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
