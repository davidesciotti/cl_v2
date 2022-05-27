import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from functools import partial

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"

wil_tot = np.load("%s/output/wil_tot.npy" %path)
wig = np.load("%s/output/wig.npy" %path)

wil_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiL.dat" %path)
wig_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiG.dat" %path)

z = np.linspace(0.001, 3, 100)

column = 9 # starts from 1, not from 0 (which is redshift)
selector = "wil"

if selector == "wil":
    # note that the first column is the redshift
#    plt.plot(wil_tot[:,0], wil_tot[:, column], "r", label = " wil davide")
    plt.plot(z, wil_tot[:, column], "r", label = " wil davide")
    plt.plot(wil_vincenzo[:,0], wil_vincenzo[:,column+1], "b", label = "wil vincenzo")
    plt.ylabel("wil")
    plt.title("WF for lensing")
    plt.legend()
else:
    plt.plot(wig[:,0], wig[:, column], "r", label = " wil davide")
    plt.plot(wig_vincenzo[:,0], wig_vincenzo[:,column], "b", label = "wil vincenzo")
    plt.ylabel("wig")
    plt.title("WF for galaxy clustering")
    plt.legend()
#plt.yscale("log")
plt.xlabel("$z$")

