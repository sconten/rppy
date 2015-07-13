import rppy
import numpy as np
import matplotlib.pyplot as plt

thetas = 1
phi = 30

p1 = 2600
chi1 = 0
C1 = rppy.reflectivity.Cij(0, 0, 0, p1, 2260, 1428)

p2 = 2700
chi2 = 0
C2 = rppy.reflectivity.Cij(0.05, 0.02, 0.1, p1, 2370, 1360)


Rphti = rppy.reflectivity.exact_ortho(C1, p1, C2, p2, chi1, chi2, phi, thetas)

print(Rphti)
