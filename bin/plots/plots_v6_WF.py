import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

def namestr(obj, namespace):
    # method to get array name
    return [name for name in namespace if namespace[name] is obj][0]

def interpolation(z, array_toInterpolate, column):
    interpolation = interp1d(array_toInterpolate[:,0], array_toInterpolate[:, column + 1], kind = "linear")
    result_array = interpolation(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result
    
def compare(array_toInterpolate, reference_array):
    # check if it's wil and shrink the array
    reference_array_name = namestr(reference_array, globals())
    # reference_array = reference_array[1:,:] # for the interpolation range
    if reference_array_name == "wil_davide" or reference_array_name == "wil_tot_davide": 
        bound = reference_array.shape[0]-1
        reference_array = reference_array[:bound,:] # for the interpolation range
    # actual method
    z = reference_array[:,0]
    interpolated_array = np.zeros((reference_array.shape[0], zbins + 1))
    interpolated_array[:,0] = z
    difference = np.zeros((reference_array.shape[0], zbins + 1))
    difference[:,0] = z
    for i in range(zbins):
        interpolated_array[:, i+1] = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in z] )    
        difference[:, i+1] = np.abs( (reference_array[:, i + 1] - interpolated_array[:, i + 1])/reference_array[:, i + 1] ) * 100
    return interpolated_array, difference

def nullify(array, bound):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.abs(array[i,j]) < bound:
                array[i,j] = 0
    return array

def nullify_vincenzo(array_davide, array_vincenzo):
    interpolated_array, difference = compare(array_vincenzo, array_davide)
    for i in range(array_davide.shape[0]):
        for j in range(array_davide.shape[1]):
            if np.abs(interpolated_array[i,j] == 0):
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
# WFs_folder = "WFs_v5_ncutVincenzo"
# WFs_folder = "WFs_v2"
# WFs_folder = "WFs_v6_zcut"
WFs_folder ="WFs_v7_zcut_noNormalization"
#################
wig_davide = np.genfromtxt(r"%s\output\%s\wig.txt" %(path, WFs_folder))
wig_SSC_davide = np.genfromtxt(r"%s\output\%s\wig_SSC.txt" %(path, WFs_folder))
wil_davide = np.genfromtxt(r"%s\output\%s\wil.txt" %(path, WFs_folder))
wil_tot_davide = np.genfromtxt(r"%s\output\%s\wil_tot.txt" %(path, WFs_folder))
# wil_SSC_davide = np.genfromtxt(r"%s\output\%s\wil_SSC.txt" %(path, WFs_folder))
niz_davide = np.genfromtxt(r"%s\output\%s\niz.txt" %(path, WFs_folder))


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
    # plt.yscale("log")
    plt.xlabel("$z$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def stacked_plots(array_davide, array_vincenzo, column):
    interpolated_array, difference = compare(array_vincenzo, array_davide)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 13))
    axes[0].plot(difference[:,0], difference[:,column+1]) 
    axes[0].set(xlabel='z', ylabel= "$\dfrac{|W_i^{L, davide}(z) - W_i^{L, vincenzo}(z)|}{W_i^{L, davide}(z)}$", title = '$\Delta W_i^L(z)/W_i^{L, davide}$ (percent), i = %i' %column)
    # axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(array_davide[:,0], array_davide[:,column+1], label = "davide") 
    axes[1].plot(array_vincenzo[:,0], array_vincenzo[:,column+1], label = "vincenzo")
    axes[1].set(xlabel='z', ylabel='$W_i^L(z)$', title = "$W_i^L(z)$, i = %i" %column)
    axes[1].legend()
    axes[1].grid()
    
    axes[2].plot(array_davide[:,0], array_davide[:,column+1], label = "davide") 
    axes[2].plot(array_vincenzo[:,0], array_vincenzo[:,column+1], label = "vincenzo")
    axes[2].set(yscale = "log", xlabel='z', ylabel='$W_i^L(z)$', title = "$W_i^L(z)$ i = %i, log scale" %column)
    axes[2].legend()
    axes[2].grid()
    
    fig.tight_layout()

column = 0
stacked_plots(wil_tot_davide, wil_vincenzo_2, column)

# interpolated_array, difference = compare(wil_vincenzo_2, wil_davide)
# interpolated_array, difference = compare(wig_vincenzo_2, wig_davide)
# interpolated_array, difference = compare(niz_vincenzo_2, niz_davide)

# plt.plot(difference[:,0], difference[:,column+1], ".-")
# plt.plot(interpolated_array[:,0], interpolated_array[:,column+1], ".-", label = "interpolated array")
# plot(wil_davide, wil_vincenzo_2, niz_sylvain, column)
# plot(wil_tot_davide, wil_vincenzo_2, niz_sylvain, column)
# plot(wig_davide, wig_vincenzo_2, niz_sylvain, column)
# plot(niz_davide, niz_vincenzo, niz_sylvain, column)




    
    
    
    