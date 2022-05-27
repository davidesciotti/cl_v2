import numpy as np
import matplotlib.pyplot as plt

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"

PS = np.load("%s/output/P(k,z).npy" %path)

h = 0.67
k_max =50/h # xxx unità di misura? * o / h? kmax??

z_array = np.linspace(0.000001, 3,   num=303)
k       = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)

column = 9 # starts from 1, not from 0 (which is redshift)

for column in [1, 50, 200, 300]:
    plt.plot(k[:450], PS[:450, column], label = "PS, z = %f" %z_array[column])
    plt.ylabel("P(k, z)")
    plt.title("Matter Power Spectrum")
    plt.legend()
    plt.grid("true", linestyle = "--")
#    plt.yscale("log")
    plt.xlabel("$k$")

