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
    theta_crit = np.arcsin(vp1/vp2)
    if theta1 >= theta_crit:
        print('Given angle exceeds critical angle for system')
    else:
        p = np.sin(theta1)/vp1      # Ray parameter
        theta2 = np.arcsin(p*vp2)   # P-wave refraction
        thetas1 = np.arcsin(p*vs1)  # S-wave reflection
        thetas2 = np.arcsin(p*vs2)  # S-wave refraction

        return(theta2, thetas1, thetas2, p)


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
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
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    theta = (theta1 + theta2) / 2.
    theta=theta2
    thetas = (thetas1 + thetas2) / 2.
    rho = (rho1 + rho2) / 2.
    vp = (vp1 + vp2) / 2.
    vs = (vs1 + vs2) / 2.

    Rpp = 0.5*(1.-4.*(p**2)*(vs**2))*drho/rho + (1/(2.*np.cos(theta)))*(dvp/vp) - 4.*(p**2)*(vs**2)*dvs/vs

    return(Rpp)


def bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    theta2, thetas1, thetas2, p = snell(vp1, vp2, vs1, vs2, theta1)

    Rpp = (1./2.)*np.log((vp2*rho2*np.cos(theta1))/(vp1*rho1*np.cos(theta2))) + \
        (np.sin(theta1)/vp1)**2*(vs1**2 - vs2**2)*(2. + np.log(rho2/rho1)/np.log(vs2/vs1))

    return Rpp


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1):

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


def youngs(bulk=None, poissons=None, mu=None, lamb=None, vp=None, vs=None, rho=None):
    return(None)


def bulk(youngs=None, poissons=None, mu=None, lamb=None, vp=None, vs=None, rho=None):
    return(None)


def poissons(youngs=None, bulk=None, mu=None, lamb=None, vp=None, vs=None, rho=None):
    return(None)


def mu(youngs=None, bulk=None, poissons=None, lamb=None, vp=None, vs=None, rho=None):
    return(None)


def lamb(youngs=None, bulk=None, poissons=None, mu=None, vp=None, vs=None, rho=None):
    return(None)


def vp(youngs=None, bulk=None, poissons=None, mu=None, lamb=None, vs=None, rho=None):
    if mu and lamb:
        return np.sqrt((lamb + 2.*mu)/rho)
    elif youngs and mu:
        return np.sqrt(mu*(youngs-4.*mu)/(rho*(youngs-3.*mu)))
    elif youngs and poissons:
        return np.sqrt(youngs*(1-poissons)/(rho*(1-poissons*(1-2.*poissons))))
    elif bulk and lamb:
        return np.sqrt((9.*bulk-2.*lamb)/rho)
    elif bulk and mu:
        return np.sqrt((bulk+4.*mu/3.)/rho)
    elif lamb and poissons:
        return np.sqrt(lamb*(1-poissons)/(poissons*rho))
    else:
        return None


def vs(youngs=None, bulk=None, poissons=None, mu=None, lamb=None, vp=None, rho=None):
    if mu and rho:
        return np.sqrt(mu/rho)
    else:
        return None


def voight_bound(M, f):
    # Compute Voight average of the input mineral given:
    # moduli M and volume fractions f
    v = np.sum(M*f)
    return (v)


def reuss_bound(M, f):
    # Compute Reuss average of the input mineral given:
    # moduli M and volume fractions f
    r = 1/np.sum(f/M)
    return (r)


def hashin_shtrikman_bounds(K, u, f):
    # Comute Hashin-Shtrikman-Walpole upper and lower bounds of a
    # multi-constituent mixture

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


def main(*args):

    K = np.array([36, 2.2])
    u = np.array([31, 2.2])

    fw = np.arange(0, 1, 0.01)

    v = np.empty(np.shape(fw))
    r = np.empty(np.shape(fw))
    hsu = np.empty(np.shape(fw))
    hsl = np.empty(np.shape(fw))

    for x in np.arange(0, len(fw)):
        v[x] = voight_bound(K, np.array([1-fw[x], fw[x]]))
        r[x] = reuss_bound(K, np.array([1-fw[x], fw[x]]))
        returned = hashin_shtrikman_bounds(K, u, np.array([1-fw[x], fw[x]]))
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
    plt.show()


if __name__ == "__main__":
    main()
