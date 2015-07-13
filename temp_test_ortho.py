import rppy
import numpy as np
import matplotlib.pyplot as plt

thetas = np.arange(1, 89, 1)
Rphti = np.zeros(np.shape(thetas))
Rp = np.zeros(np.shape(thetas))
phi = 45

p1 = 2600
vp1 = 2260
vs1 = 1428
chi1 = 0
e1 = 0
d1 = 0
y1 = 0
C1 = rppy.reflectivity.Cij(e1, d1, y1, p1, vp1, vs1)

p2 = 2700
vp2 = 2370
vs2 = 1360
chi2 = 0
e2 = 0.0
d2 = 0.0
y2 = 0.0
C2 = rppy.reflectivity.Cij(e2, d2, y2, p1, vp2, vs2)

for n in thetas:
    Rphti[n-1] = rppy.reflectivity.exact_ortho(C1, p1, C2, p2, chi1, chi2, phi, thetas[n-1])

Rp = rppy.reflectivity.zoeppritz(vp1, vs1, p1, vp2, vs2, p2, np.radians(thetas))
Rp_rug = rppy.reflectivity.ruger_hti(vp1, vs1, p1, e1, d1, y1, vp2, vs2, p2, e2, d2, y2, np.radians(thetas), np.radians(phi))

print(Rphti)
plt.figure(1)
plt.plot(thetas, Rphti, thetas, Rp, thetas, Rp_rug)
plt.show()