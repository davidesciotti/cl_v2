import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

def interpolation(z, array_toInterpolate, column):
    interpolation = interp1d(array_toInterpolate[:,0], array_toInterpolate[:,column + 1], kind = "linear")
    result_array = interpolation(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

####################################### function definition
c = 299792.458 # km/s 

zbins = 10
zmin = 0.01
zmax = 2
zsteps = 100
z = np.linspace(zmin, zmax, zsteps)
# davide ##########################################
wig_davide = np.load(r"%s\output\base_functions_v1\wig(z,i).npy" %path)
wig_SSC_davide = np.load(r"%s\output\base_functions_v1\wig_SSC(z,i).npy" %path)

wil_davide = np.load(r"%s\output\base_functions_v1\wil(z,i).npy" %path)
wil_tot_davide = np.load(r"%s\output\base_functions_v1\wil_tot(z,i).npy" %path)
wil_SSC_davide = np.load(r"%s\output\base_functions_v1\wil_SSC(z,i).npy" %path)

niz_davide = np.load(r"%s\output\base_functions_v1\n_i(z,i).npy" %path)
# Dz_davide = np.load(r"%s\output\base_functions_v1\D(z).npy" %path)

wig_davide_5 = np.genfromtxt(r"%s\output\prove_varie\wig_SSC_davide_5.txt" %path)


# vincenzo old ##############################
wil_SSC_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\WiL.dat" %path)
wig_SSC_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\WiG.dat" %path)
niz_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\niNorm.dat" %path)
Dz_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\GrowthFactor.dat" %path)
#vincenzo new
niz_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\niTab-EP10-RB00.dat" %path)
wil_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\WiGamma-LCDM-NonLin-eNLA.dat" %path)
wig_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\WiG-LCDM-NonLin-eNLA.dat" %path)

# sylvain
z_sylvain = np.genfromtxt(r"%s\data\windows_sylvain\nz_source\z.txt" %path)
wil_sylvain = np.zeros((z_sylvain.shape[0], 11))
wig_sylvain = np.zeros((z_sylvain.shape[0], 11))
niz_sylvain  = np.zeros((z_sylvain.shape[0], 11))

wil_sylvain[:,0] = wig_sylvain[:,0] = niz_sylvain[:,0] = z_sylvain

for i in range(10):
    wil_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\IST_kernels\W_WL_%i.txt" %(path, i+1))
    wig_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\IST_kernels\W_GC_%i.txt" %(path, i+1))
    niz_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\nz_source\bin_%i.txt" %(path, i+1))


# plotting




def plot(array_1, array_2, array_3, column):
    # column starts from 1, not from 0 (which is redshift)
    # plt.plot(array_1[:,0], array_1[:,column+1], ".-", label = "davide")
    # plt.plot(array_2[:,0], array_2[:,column+1], ".-", label = "vincenzo" )
    plt.plot(array_3[:,0], array_3[:,column+1], label = "sylvain")
    
    plt.plot(array_3[:,0], array_3[:,column+1], label = "sylvain")

    # plt.plot(array_2[:,0], interpolated_array, label = "sylvain")
    plt.title("$n(z)$, bin %i" %column)
    plt.ylabel("$n_%i(z)$" %column)
#    plt.yscale("log")
    plt.xlabel("$z$")
    plt.legend()
    plt.show()


column = 0

array_toInterpolate = wig_davide
reference_array = wig_vincenzo_2[1:,:]

interpolated_array = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in reference_array[:,0]] )

difference = np.zeros((reference_array.shape[0], zbins + 1))
difference[:,0] = reference_array[:,0]
difference[:,column+1] = (reference_array[:, column + 1] - interpolated_array)/reference_array[:, column + 1] * 100

plot(wig_davide, wig_vincenzo_2, difference, column)



    
    
    
    
    
    