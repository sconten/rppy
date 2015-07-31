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
import matplotlib.pyplot as plt


def main(*args):
    import media
    import reflectivity

    K = np.array([36, 2.2])
    u = np.array([31, 2.2])

    fw = np.arange(0, 1, 0.01)

    v = np.empty(np.shape(fw))
    r = np.empty(np.shape(fw))
    h = np.empty(np.shape(fw))
    hsu = np.empty(np.shape(fw))
    hsl = np.empty(np.shape(fw))

    for x in np.arange(0, len(fw)):
        v[x], r[x], h[x] = media.voight_reuss_hill(K, np.array([1-fw[x], fw[x]]))
        returned = media.hashin_shtrikman(K, u, np.array([1-fw[x], fw[x]]))
        hsu[x] = returned[0]
        hsl[x] = returned[1]

    plt.plot(fw, v, 'k')
    plt.plot(fw, r, 'k')
    plt.plot(fw, hsu, 'k')
    plt.plot(fw, hsl, 'k')

    C1 = np.zeros(shape=(6, 6))
    C1 = [[15.12e9, 5.29e9, 6.26e9, 0,      0,      0],
          [5.29e9, 10.89e9, 6.46e9, 0,      0,      0],
          [6.26e9,  6.46e9, 9.36e9, 0,      0,      0],
          [0,       0,      0,      2.00e9, 0,      0],
          [0,       0,      0,      0,      2.09e9, 0],
          [0,       0,      0,      0,      0,      4.26e9]]

    C2 = np.zeros(shape=(6, 6))
    C2 = [[28.52e9,  7.70e9,  6.00e9, 0,      0,      0],
          [ 7.70e9, 15.21e9,  7.65e9, 0,      0,      0],
          [ 6.00e9,  7.65e9, 10.65e9, 0,      0,      0],
          [ 0,       0,       0,      2.23e9, 0,      0],
          [ 0,       0,       0,      0,      2.41e9, 0],
          [ 0,       0,       0,      0,      0,      5.71e9]]

    plt.axis([0, 1, 0, 40])
    plt.show()

    thetas = np.arange(1, 47, 1)
    Rppz = np.empty(np.shape(thetas))
    Rppb = np.empty(np.shape(thetas))
    Rppak = np.empty(np.shape(thetas))
    Rpps = np.empty(np.shape(thetas))
    Rpvti = np.empty(np.shape(thetas))
    Rphti = np.empty(np.shape(thetas))

    plt.figure(2)
    for n in range(np.size(thetas)):
        dummy = reflectivity.zoeppritz(3000, 1500,
                                       2000, 4000,
                                       2000, 2200,
                                       np.radians(thetas[n]))
        Rppz[n] = dummy[0]
        Rppb[n] = reflectivity.bortfeld(3000, 1500,
                                        2000, 4000,
                                        2000, 2200,
                                        np.radians(thetas[n]))
        Rppak[n] = reflectivity.aki_richards(3000, 1500,
                                             2000, 4000,
                                             2000, 2200,
                                             np.radians(thetas[n]))
        Rpps[n] = reflectivity.shuey(3000, 1500, 2000,
                                     4000, 2000, 2200,
                                     np.radians(thetas[n]))
        Rpvti[n] = reflectivity.ruger_vti(3000, 1500, 2000, 0.0, 0.0, 0.0,
                                          4000, 2000, 2200, 0.1, 0.1, 0.1,
                                          np.radians(thetas[n]))
        Rphtt[n] = reflectivity.exact_ortho(C1, 1400, C2, 1840, 0, 0, 30, 40)

    plt.plot(thetas, Rppz, thetas, Rppb, thetas, Rppak, thetas, Rpps, thetas, Rpvti, thetas, Rphti)
    plt.legend(['Zoeppritz', 'Bortfeld', 'Aki-Richards', 'Shuey', 'Ruger VTI', 'Schoenberg'])
    plt.xlim([0, 50])
    plt.ylim([0.15, 0.25])
    plt.show()

    t = np.arange(0, 15, 0.1)

    A = util.tuning_wedge(1, 90, t)

    plt.figure(3)
    plt.plot(A)

    #########################################
    Km = 37
    um = 44
    Ki = 2.25
    ui = 0
    xi = 0.1
    si = 'sphere'

    em = media.kuster_toksoz(Km, um, Ki, ui, xi, si)

    print(em['K'])
    print(em['u'])


if __name__ == "__main__":
    main()
