# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:24:04 2015

@author: Sean
"""
import rppy
import numpy as np
import matplotlib.pyplot as plt

p1 = 2000
vp1 = 3000
vs1 = 1500
e1 = 0.0
d1 = 0.0
y1 = 0.0

p2 = 2200
vp2 = 4000
vs2 = 2000
y2 = 0.1
d2 = 0.1
e2 = 0.1

theta = 30

phi = np.arange(0, 90, 1)

phit = np.array([1.2500, 4.9342, 8.6184, 11.842, 15.526, 19.211, 22.664,
                25.888, 28.421, 30.724, 34.638, 38.092, 41.546, 45.461,
                49.375, 53.289, 56.974, 60.888, 65.493, 69.408, 73.783,
                79.079, 84.375, 89.211])
exp = np.array([0.19816, 0.19816, 0.19678, 0.19539, 0.19263, 0.19056,
                0.18711, 0.18365, 0.18020, 0.17813, 0.17329, 0.16845,
                0.16431, 0.15878, 0.15326, 0.14842, 0.14359, 0.13875,
                0.13391, 0.12977, 0.12632, 0.12286, 0.12079, 0.12010])
Rpp = np.zeros(np.shape(phi))
Rpo = np.zeros(np.shape(phi))
Rpk = np.zeros(np.shape(phi))

for ind, phiv in enumerate(phi):
    Rpp[ind] = rppy.reflectivity.ruger_hti(vp1, vs1, p1, e1, d1, y1,
                                           vp2, vs2, p2, e2, d2, y2,
                                           theta, phiv)
    Rpo[ind] = rppy.reflectivity.exact_ortho(rppy.reflectivity.Cij(vp1, vs1, p1, 0, 0, 0, e1, d1, y1, 0), p1,
                                             rppy.reflectivity.Cij(vp2, vs2, p2, 0, 0, 0, e2, d2, y2, 0), p2,
                                             0, 0, phiv, theta)
    Rpk[ind] = rppy.reflectivity.vavrycuk_psencik_hti(vp1, vs1, p1, e1, d1, y1,
                                                      vp2, vs2, p2, e2, d2, y1,
                                                      phiv, theta)


plt.figure(1)
plt.plot(phi, Rpp, phi, Rpo, phi, Rpk)
plt.show()

theta = np.arange(0, 60, 1)
phi = 45

Rpp = np.zeros(np.shape(theta))
Rpo = np.zeros(np.shape(theta))
Rpk = np.zeros(np.shape(theta))
Rpa = np.zeros(np.shape(theta))

for ind, thetav in enumerate(theta):
    Rpp[ind] = rppy.reflectivity.ruger_hti(vp1, vs1, p1, e1, d1, y1,
                                           vp2, vs2, p2, e2, d2, y1,
                                           thetav, phi)
    Rpk[ind] = rppy.reflectivity.vavrycuk_psencik_hti(vp1, vs1, p1, e1, d1, y1,
                                                      vp2, vs2, p2, e2, d2, y1,
                                                      phi, thetav)

Rpo = rppy.reflectivity.zoeppritz(vp1, vs1, p1, vp2, vs2, p2, theta)
Rpa = rppy.reflectivity.aki_richards(vp1, vs1, p1, vp2, vs2, p2, theta)

plt.figure(2)
plt.plot(theta, Rpp, theta, Rpo, theta, Rpk, theta, Rpa)
plt.xlim([0, 60])
plt.ylim([0.125, 0.275])
plt.legend(['Ruger', 'Zoe', 'Vavrycuk', 'A-R'])
plt.show()
