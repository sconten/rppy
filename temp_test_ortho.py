import rppy
import numpy as np
import matplotlib.pyplot as plt

thetas = np.arange(1, 89, 1)
Rphti = np.zeros(np.shape(thetas))
Rp = np.zeros(np.shape(thetas))
phi = 45

vp1 = 2260
vs1 = 1428
p1 = 2600
chi1 = 0
C1 = np.array([[1.32797600e+10, 2.67601320e+09, 2.67600320e+09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [2.67600320e+09, 1.32797600e+10, 2.67600320e+09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [2.67600320e+09, 2.67600320e+09, 1.32797600e+10, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.30187840e+09, 0.00000000e+00, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.30187840e+09, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.30187840e+09]])
vp1, vs1, e11, d11, y11, e21, d21, y21, d31 = rppy.reflectivity.thomsen(C1, p1)
Cbar = rppy.reflectivity.Cij(vp1, vs1, p1, e11, d11, y11, e21, d21, y21, d31)

p2 = 2700
chi2 = 0
C2 = np.array([[1.46039400e+10, 4.98602000e+09, 4.98602000e+09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [4.98602000e+09, 1.46039400e+10, 4.98601000e+09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [4.98602000e+09, 4.98602000e+09, 1.46039400e+10, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80896000e+09, 0.00000000e+00, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80896000e+09, 0.00000000e+00],
               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80896000e+09]])
vp2, vs2, e12, d12, y12, e22, d22, y22, d32 = rppy.reflectivity.thomsen(C2, p2)

for n in thetas:
    Rphti[n-1] = rppy.reflectivity.exact_ortho(C1, p1, C2, p2, chi1, chi2, phi, thetas[n-1])

Rp = rppy.reflectivity.zoeppritz(vp1, vs1, p1, vp2, vs2, p2, np.radians(thetas))
Rp_rug = rppy.reflectivity.ruger_hti(vp1, vs1, p1, e21, d21, y21, vp2, vs2, p2, e22, d22, y22, np.radians(thetas), np.radians(phi))

plt.figure(1)
plt.plot(thetas, Rphti, thetas, Rp, thetas, Rp_rug)
plt.legend(['HTI', 'Zoe', 'Ruger'])
plt.xlim([10, 70])
plt.ylim([0, 0.2])
plt.show()
np.set_printoptions(precision=2)
print()
print('Original C')
print(C1)
print()
print('Paras')
print(vp1, vs1, p1, e11, d11, y11, e21, d21, y21, d31)
print()
print('New C')
print(Cbar)