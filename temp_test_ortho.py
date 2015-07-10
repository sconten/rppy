import rppy
import numpy as np
import matplotlib.pyplot as plt

phi = np.radians(30)
thetas = np.array([40])
Rphti = np.empty(np.shape(thetas))

p1 = 1400
chi1 = 0
C1 = np.zeros(shape=(6, 6))
C1 = [[15.12e9, 5.29e9, 6.26e9, 0,      0,      0],
      [5.29e9, 10.89e9, 6.46e9, 0,      0,      0],
      [6.26e9,  6.46e9, 9.36e9, 0,      0,      0],
      [0,       0,      0,      2.00e9, 0,      0],
      [0,       0,      0,      0,      2.09e9, 0],
      [0,       0,      0,      0,      0,      4.26e9]]

p2 = 1840
chi2 = 0
C2 = np.zeros(shape=(6, 6))
C2 = [[28.52e9,  7.70e9,  6.00e9, 0,      0,      0],
      [ 7.70e9, 15.21e9,  7.65e9, 0,      0,      0],
      [ 6.00e9,  7.65e9, 10.65e9, 0,      0,      0],
      [ 0,       0,       0,      2.23e9, 0,      0],
      [ 0,       0,       0,      0,      2.41e9, 0],
      [ 0,       0,       0,      0,      0,      5.71e9]]
      
for n in range(np.size(thetas)):
    Rphti[n] = rppy.reflectivity.exact_ortho(C1, 1400, C2, 1840, 0, 0, 30, thetas[n])

plt.figure(1)
plt.plot(thetas, Rphti)
plt.show()