import rppy
import numpy as np
import matplotlib.pyplot as plt

vp1 = 3000
vs1 = 1500
p1 = 2000
e1_1 = 0.0
d1_1 = 0.0
y1_1 = 0.0
e2_1 = 0.0
d2_1 = 0.0
y2_1 = 0.0
d3_1 = 0.0
chi1 = 0.0
C1 = rppy.reflectivity.Cij(vp1, vs1, p1, e1_1, d1_1, y1_1, e2_1, d2_1, y2_1, d3_1)

vp2 = 4000
vs2 = 2000
p2 = 2200
e1_2 = 0.0
d1_2 = 0.0
y1_2 = 0.0
e2_2 = 0.0
d2_2 = 0.0
y2_2 = 0.0
d3_2 = 0.0
chi2 = 0.0
C2 = rppy.reflectivity.Cij(vp2, vs2, p2, e1_2, d1_2, y1_2, e2_2, d2_2, y2_2, d3_2)

phi = np.arange(0, 90, 1)
theta = np.arange(0, 90, 1)

loopang = phi
theta = np.array([30])

rphti = np.zeros(np.shape(loopang))
rpzoe = np.zeros(np.shape(loopang))
rprug = np.zeros(np.shape(loopang))

for aid, val in enumerate(loopang):
    rphti[aid] = rppy.reflectivity.exact_ortho(C1, p1, C2, p2, chi1, chi2, loopang[aid], theta)
    rprug[aid] = rppy.reflectivity.ruger_hti(vp1, vs1, p1, e2_1, d2_1, y2_1, vp2, vs2, p2, e2_2, d2_2, y2_2, np.radians(theta), np.radians(loopang[aid]))
    rpzoe[aid] = rppy.reflectivity.zoeppritz(vp1, vs1, p1, vp2, vs2, p2, np.radians(theta))

plt.figure(1)
plt.plot(loopang, rphti, loopang, rprug, loopang, rpzoe)
plt.legend(['hti', 'ruger', 'zoe'])
plt.show()