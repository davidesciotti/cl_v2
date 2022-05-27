import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

def interpolation(z, array_toInterpolate, column):
    interpolation = interp1d(array_toInterpolate[:,0], array_toInterpolate[:, column + 1], kind = "linear")
    result_array = interpolation(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result
    
def compare_wig(array_toInterpolate, reference_array):
    # reference_array = reference_array[1:,:] # for the interpolation range
    z = reference_array[:,0]
    interpolated_array = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in z] )    
    difference = np.zeros((reference_array.shape[0], zbins + 1))
    difference[:,0] = z
    difference[:, column+1] = np.abs( (reference_array[:, column + 1] - interpolated_array)/reference_array[:, column + 1] ) * 100
    return interpolated_array, difference

def compare_wil(array_toInterpolate, reference_array):
    reference_array = reference_array[:299,:] # for the interpolation range
    z = reference_array[:,0]
    interpolated_array = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in z] )    
    difference = np.zeros((reference_array.shape[0] , zbins + 1))
    difference[:,0] = z
    difference[:, column+1] = np.abs( (reference_array[:, column + 1] - interpolated_array)/reference_array[:, column + 1] ) * 100
    return interpolated_array, difference

def nullify(array, bound):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.abs(array[i,j]) < bound:
                array[i,j] = 0
    return array

def nullify_vincenzo_wil(array_davide, array_vincenzo):
    interpolated_array, difference = compare_wil(array_vincenzo, array_davide)
    for i in range(array_davide.shape[0]):
        for j in range(array_davide.shape[1]):
            if np.abs(array_vincenzo[i,j] == 0):
                array_davide[i,j] = 0
    return array_davide

def nullify_vincenzo_wig(array_davide, array_vincenzo):
    interpolated_array, difference = compare_wil(wil_vincenzo_2, wil_davide)
    for i in range(array_davide.shape[0]):
        for j in range(array_davide.shape[1]):
            if np.abs(array_vincenzo[i,j] == 0):
                array_davide[i,j] = 0
    return array_davide
####################################### function definition
c = 299792.458 # km/s 

zbins = 10
zmin = 0.001
zmax = 4
zsteps = 100
z = np.linspace(zmin, zmax, zsteps)
############################### DAVIDE ##########################################
# wig_davide = np.load(r"%s\output\WFs_v3_cut\wig.npy" %path)
# wig_SSC_davide = np.load(r"%s\output\WFs_v3_cut\wig_SSC.npy" %path)
# wil_davide = np.load(r"%s\output\WFs_v3_cut\wil.npy" %path)
# wil_tot_davide = np.load(r"%s\output\WFs_v3_cut\wil_tot.npy" %path)
# # wil_SSC_davide = np.load(r"%s\output\WFs_v3_cut\wil_SSC.npy" %path)
# niz_davide = np.genfromtxt(r"%s\output\WFs_v3_cut\niz_e-19cut.txt" %path)

# con niz vincenzo
# wig_davide = np.load(r"%s\output\WFs_v4_nizVincenzo\wig.npy" %path)
# wig_SSC_davide = np.load(r"%s\output\WFs_v4_nizVincenzo\wig_SSC.npy" %path)
# wil_tot_davide = np.load(r"%s\output\WFs_v4_nizVincenzo\wil_tot.npy" %path)

# quelle normali
wig_davide = np.load(r"%s\output\WFs_v2\wig(z,i).npy" %path)
wig_SSC_davide = np.load(r"%s\output\WFs_v2\wig_SSC(z,i).npy" %path)
wil_davide = np.load(r"%s\output\WFs_v2\wil(z,i).npy" %path)
wil_tot_davide = np.load(r"%s\output\WFs_v2\wil_tot(z,i).npy" %path)
wil_SSC_davide = np.load(r"%s\output\WFs_v2\wil_SSC(z,i).npy" %path)
niz_davide = np.load(r"%s\output\WFs_v2\n_i(z,i).npy" %path)

############################### VINCENZO OLD ##########################################
wil_SSC_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\WiL.dat" %path)
wig_SSC_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\WiG.dat" %path)
niz_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\niNorm.dat" %path)
Dz_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\GrowthFactor.dat" %path)
############################### VINCENZO NEW ##########################################
niz_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\niTab-EP10-RB00.dat" %path)
wil_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\WiGamma-LCDM-NonLin-eNLA.dat" %path)
wig_vincenzo_2 = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\WiG-LCDM-NonLin-eNLA.dat" %path)

############################### SYLVAIN ##########################################
z_sylvain = np.genfromtxt(r"%s\data\windows_sylvain\nz_source\z.txt" %path)
wil_sylvain = np.zeros((z_sylvain.shape[0], 11))
wig_sylvain = np.zeros((z_sylvain.shape[0], 11))
niz_sylvain  = np.zeros((z_sylvain.shape[0], 11))

wil_sylvain[:,0] = wig_sylvain[:,0] = niz_sylvain[:,0] = z_sylvain

for i in range(10):
    wil_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\IST_kernels\W_WL_%i.txt" %(path, i+1))
    wig_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\IST_kernels\W_GC_%i.txt" %(path, i+1))
    niz_sylvain[:,i+1] = np.genfromtxt(r"%s\data\windows_sylvain\nz_source\bin_%i.txt" %(path, i+1))

# wil_tot_davide = nullify(wil_tot_davide)
# wig_davide = nullify(wig_davide)
# wil_SSC_davide = nullify(wil_SSC_davide)
# wig_SSC_davide = nullify(wig_SSC_davide)
# niz_davide = nullify(niz_davide)

def plot(array_1, array_2, array_3, column):
    # column starts from 1, not from 0 (which is redshift)
    plt.plot(array_1[:,0], array_1[:,column+1], ".-", label = "davide")
    plt.plot(array_2[:,0], array_2[:,column+1], ".-", label = "vincenzo" )
    # plt.plot(array_3[:,0], array_3[:,column+1], label = "sylvain")
    
    plt.title("$n(z)$, bin %i" %column)
    plt.ylabel("$n_%i(z)$" %column)
    plt.yscale("log")
    plt.xlabel("$z$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
column = 8


# wil_davide = nullify(wil_davide, 1e-25)
# wig_davide = nullify(wig_davide, 1e-20)

interpolated_array, difference = compare_wil(wil_vincenzo_2, wil_davide)
# interpolated_array, difference = compare_wig(wig_vincenzo_2, wig_davide)
# interpolated_array, difference = compare_wil(niz_vincenzo_2, niz_davide)


# plt.plot(difference[:,0], difference[:,column+1], ".-")
# plt.plot(difference[:,0], interpolated_array, ".-", label = "interpolated array")
# plot(wil_davide, wil_vincenzo_2, niz_sylvain, column)
plot(wig_davide, wig_vincenzo_2, niz_sylvain, column)
# plot(niz_davide, niz_vincenzo, niz_sylvain, column)

np.savetxt(r"%s\output\wil_davide.txt" %path, wil_davide[:,:2])
np.savetxt(r"%s\output\wil_vincenzo_2.txt" %path, wil_vincenzo_2[:,:2])
np.savetxt(r"%s\output\interpolated_array.txt" %path, interpolated_array)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
# axes[0].plot(difference[:,0], difference[:,column+1], ".-", label = "difference") 
# axes[1].plot(wil_davide[:,0], wil_davide[:,column+1], ".-", label = "wil_davide") 
# axes[1].plot(wil_vincenzo_2[:,0], wil_vincenzo_2[:,column+1], ".-", label = "wil_vincenzo_2")
# fig.tight_layout()

# plt.plot(wig_vincenzo_2[:,0], wig_vincenzo_2[:,1], label = "vincenzo original") 
# plt.plot(wig_davide[:,0], interpolated_array, label = "interpolated_array") 
# plt.legend()

# array = niz_vincenzo
# for i in range(array.shape[0]):
#     # for j in range(wig_vincenzo_2.shape[1]):
#     if array[i,1] < 1e-18 and array[i,1] != 0:
#         print(array[i,1])




    
    
    
    
    
    