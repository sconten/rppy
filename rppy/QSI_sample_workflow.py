import numpy
import rppy
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import las

# This script demonstrates a sample workflow from chapter 7 of the
# Quantitative Seismic Interpretation by Avseth, Mkerji, and Mavko.


def plotlog(depth, curve, cname, clr, axmin, axmax, dmin, dmax):
    plt.plot(curve, depth, clr)
    plt.ylim(2000, 2600)
    plt.title(cname)
    plt.xlim(axmin, axmax)
    plt.gca().set_xticks([axmin, axmax])
    plt.gca().xaxis.grid(True, which="minor")
    minorLoc = AutoMinorLocator(6)
    plt.gca().xaxis.set_minor_locator(minorLoc)
    plt.gca().invert_yaxis()


def rho2phi(rho_m, rho_f, rho_b):
    phi = (rho_b - rho_m)/(rho_f - rho_m)
    return phi


def main(*args):
    # Parameters

    # Load log data for well #2
    well2 = las.LASReader("well_2.las", null_subs=numpy.nan)

    # Plot logs

    plt.figure(1)
    plt.suptitle("Well #2 Log Suite")

    plt.subplot(1, 4, 1)
    plotlog(well2.data["DEPT"], well2.data["GR"], "GR", "g", 20, 120, 2100, 2500)
    plt.ylabel("Depth [m]")
    plt.gca().set_yticks([2100, 2200, 2300, 2400, 2500])

    plt.subplot(1, 4, 2)
    plotlog(well2.data["DEPT"], well2.data["RHOB"], "RHOB", "m", 1.65, 2.65, 2100, 2500)
    plt.gca().axes.get_yaxis().set_ticks([])

    plt.subplot(1, 4, 3)
    plotlog(well2.data["DEPT"], well2.data["Vp"], "Vp", "b", 0.5, 4.5, 2100, 2500)
    plotlog(well2.data["DEPT"], well2.data["Vs"], "VELOCITY", "r", 0.5, 4.5, 2100, 2500)
    plt.gca().axes.get_yaxis().set_ticks([])

    # Compute density assuming a mineral density of 2.65 and a fluid density of 1.05
    phi = rho2phi(2.65, 1.05, well2.data["RHOB"])

    # Plot it with other logs
    plt.subplot(1, 4, 4)
    plotlog(well2.data["DEPT"], phi, "POR", 'k', 0, 0.6, 2100, 2500)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()

    # Cross-plots of Vp vs. Porosity
    plt.figure(2)
    plt.scatter(phi, well2.data["Vp"], s=40, c=well2.data["GR"], alpha=0.5)
    plt.xlabel('POROSITY')
    plt.ylabel('VP')
    plt.show()

    # Compute Hashin-Shtrikman bounds

if __name__ == "__main__":
    main()
