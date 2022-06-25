import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"
z = np.linspace(0.01, 4, 45)
################# davide
wil_davide = np.load("%s/output/night_computation_zColumn/wil_tot.npy" %path)
wig_davide = np.load("%s/output/night_computation_zColumn/wig.npy" %path)
wil_SSC_davide = np.load("%s/output/night_computation_zColumn/wil_SSC.npy" %path)
wig_SSC_davide = np.load("%s/output/night_computation_zColumn/wig_SSC.npy" %path)
niz_davide = np.load("%s/output/night_computation_zColumn/n_i(z).npy" %path)
Dz_davide = np.load("%s/output/night_computation_zColumn/D(z).npy" %path)

wig_davide_5 = np.genfromtxt("%s/output/prove_varie/wig_SSC_davide_5.txt" %path)


################# vincenzo old
wil_SSC_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiL.dat" %path)
wig_SSC_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiG.dat" %path)
niz_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/niNorm.dat" %path)
Dz_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/GrowthFactor.dat" %path)
######## vincenzo new
niz_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path)
wil_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/WiGamma-LCDM-NonLin-eNLA.dat" %path)
wig_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/WiG-LCDM-NonLin-eNLA.dat" %path)

################# sylvain
z_sylvain = np.genfromtxt("%s/data/windows_sylvain/nz_source/z.txt" %path)
wil_sylvain = np.zeros((z_sylvain.shape[0], 11))
wig_sylvain = np.zeros((z_sylvain.shape[0], 11))
niz_sylvain  = np.zeros((z_sylvain.shape[0], 11))

wil_sylvain[:,0] = wig_sylvain[:,0] = niz_sylvain[:,0] = z_sylvain

for i in range(10):
    wil_sylvain[:,i+1] = np.genfromtxt("%s/data/windows_sylvain/IST_kernels/W_WL_%i.txt" %(path, i+1))
    wig_sylvain[:,i+1] = np.genfromtxt("%s/data/windows_sylvain/IST_kernels/W_GC_%i.txt" %(path, i+1))
    niz_sylvain[:,i+1] = np.genfromtxt("%s/data/windows_sylvain/nz_source/bin_%i.txt" %(path, i+1))


############ plotting
c = 299792.458 # km/s is the unit correct??

def plot(array_1, array_2, array_3, column):
    # column starts from 1, not from 0 (which is redshift)
    plt.plot(array_1[:,0], array_1[:,column+1], label = "old")
    plt.plot(array_2[:,0], array_2[:,column+1], label = "new")
#    plt.plot(array_3[:,0], array_3[:,column+1], label = "sylvain")

    plt.title("$n(z)$, bin %i" %column)
    plt.ylabel("$n_%i(z)$" %column)
#    plt.yscale("log")
    plt.xlabel("$z$")
    plt.legend()

plot(wil_davide, wil_vincenzo_2, niz_sylvain, 9)
#plot(niz_davide, niz_vincenzo, niz_sylvain, 2)
#plt.plot(z, wig_davide_5, label = "new") 
#plt.legend()
    
    
    
    
    
    
    
    
    