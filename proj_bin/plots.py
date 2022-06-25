import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16


path = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide"

def interpolation(z, array_toInterpolate, column):
    interpolation = interp1d(array_toInterpolate[:,0], array_toInterpolate[:,column + 1], kind = "linear")
    result_array = interpolation(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

def namestr(obj, namespace):
    # method to get array name
    return [name for name in namespace if namespace[name] is obj][0]

def plot(array_1, array_2, i, j):
    plt.plot(ell_values, array_1[:,i,j], ".-", label = "davide")
    plt.plot(ell_values, array_2[:,i,j], ".-", label = "vincenzo" )
    
    plt.title("$Cij(\ell)$")
    plt.ylabel("$Cij(\ell)$")
    plt.yscale("log")
    plt.xlabel("$\ell$")
    plt.legend()
    plt.show()

def compare(array_1, array_2, i, j):
    # xxx remember that it's dangerous to equalize floats
    print(array_1[:,0,0])
    print(array_2[:,0,0])
    print(np.isclose(array_1, array_2))
    if np.all(array_1) == np.all(array_2): print("the arrays are equal")
    if np.all(array_1) == 0 or np.all(array_2) == 0: print("one of the arrays is null")
    diff = np.abs(array_1[:,i,j] - array_2[:,i,j]) / array_1[:,i,j] * 100
    # print(array_1[:,i,j], array_2[:,i,j])
    return diff

def stacked_plots(array_davide, array_vincenzo, row, column):
    difference = compare(array_vincenzo, array_davide, i, j)
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 13))
    axes[0].plot(ell_values, difference, ".-") 
    axes[0].set(xlabel='$\ell$', ylabel= "$\dfrac{|C_{ij}^{GG, davide}(\ell) - C_{ij}^{GG, vincenzo}(\ell)|}{C_{ij}^{GG, davide}(\ell)}$", title = '$\Delta C_{ij}^{GG}(\ell)/C_{ij}^{GG, davide}(\ell)$ (percent), i = %i, j = %i' %(i,j))
    # axes[0].legend(prop={'size': 14})
    axes[0].grid()
    
    axes[1].plot(ell_values, array_davide[:,i,j], ".-", label = "davide") 
    axes[1].plot(ell_values, array_vincenzo[:,i,j], ".-", label = "vincenzo")
    axes[1].set(xlabel='$\ell$', ylabel='$C_{ij}^{GG}(\ell)$', title = "$C_{ij}^{GG}(\ell)$, i = %i, j = %i" %(i,j))
    axes[1].legend(prop={'size': 14})
    axes[1].grid()
    
    axes[2].plot(ell_values, array_davide[:,i,j], ".-", label = "davide") 
    axes[2].plot(ell_values, array_vincenzo[:,i,j], ".-", label = "vincenzo")
    axes[2].set(yscale = "log", xlabel='$\ell$', ylabel='$C_{ij}^{GG}(\ell)$', title = "$C_{ij}^{GG}(\ell)$, i = %i, j = %i" %(i,j))
    axes[2].legend(prop={'size': 14})
    axes[2].grid()
        
    fig.tight_layout()

def diff_new(array_1, array_2):
    result = (array_1-array_2)/array_1 *100
    return result
####################################### function definition
c = 299792.458 # km/s 

zbins = 10
zmin = 0.001
zmax = 4
zsteps = 3000
z = np.linspace(zmin, zmax, zsteps)

# davide ##########################################
Cij_folder = "Cijs_v12_newell_IA"
C_LL_davide = np.load("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide = np.load("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide = np.load("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_folder))

# davide newBias ##########################################
Cij_folder = "Cijs_v13_newell_IA_newBias"
C_LL_davide_newBias = np.load("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide_newBias = np.load("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide_newBias = np.load("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_folder))

# davide WFsylvain
Cij_folder = "Cijs_v18_oldBias_WFsylvain"
C_LL_davide_WFsylvain = np.load("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide_WFsylvain = np.load("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide_WFsylvain = np.load("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_folder))

# davide WFsylvain
Cij_folder = "Cij_WFvincenzo_noIA_oldBias_nz300"
C_LL_davide_WFvincenzo = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide_WFvincenzo = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide_WFvincenzo = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_GG.npy" %(path, Cij_folder))

# davide WF davide
Cij_folder = "Cij_WFdavide_IA_newBias_nz300"
C_LL_davide_WFdavide_IA_newBias = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide_WFdavide_IA_newBias = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide_WFdavide_IA_newBias = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_GG.npy" %(path, Cij_folder))

# davide WF davide v12
Cij_folder = "Cijs_v12_newell_IA_testing"
C_LL_davide_WFdavide_IA_newBias_test = np.load("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_folder))
C_LG_davide_WFdavide_IA_newBias_test = np.load("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_folder))
C_GG_davide_WFdavide_IA_newBias_test = np.load("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_folder))

# davide WF davide optimised
# Cij_folder = "Cij_WFdavide_IA_oldBias_nz300_optimised"
# CLL_WFdavide_IA_oldBias_nz300_optimised = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LL_optimised.npy" %(path, Cij_folder))

# davide v19 NON optimised
Cij_folder = "Cij_WFdavide_IA_oldBias_nz300_testing"
Cij_WFdavide_IA_oldBias_nz300_testing = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LL.npy" %(path, Cij_folder))

# davide v19 optimised
Cij_folder = "Cij_WFdavide_IA_newBias_nz10000"
CLL_WFdavide_IA_newBias_nz10000 = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_LL.npy" %(path, Cij_folder))
CGG_WFdavide_IA_newBias_nz10000 = np.load("%s/output/Cij/Cijs_v19_ALL/%s/Cij_GG.npy" %(path, Cij_folder))


# vincenzo ##############################
C_LL_vincenzo_raw = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat" %path)
C_LG_vincenzo_raw = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat" %path)
C_GG_vincenzo_raw = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat" %path)
# anzi, prendo quelli reshaped
C_LL_vincenzo = np.load("%s/data/Cij-NonLin-eNLA_15gen/Cij_reshape/C_LL_nbl20.npy" %path)
C_LG_vincenzo = np.load("%s/data/Cij-NonLin-eNLA_15gen/Cij_reshape/C_LG_nbl20.npy" %path)
C_GG_vincenzo = np.load("%s/data/Cij-NonLin-eNLA_15gen/Cij_reshape/C_GG_nbl20.npy" %path)
# 14_may
path_new = "C:/Users/dscio/Documents/Lavoro/Programmi/master_unified_forSSCcomparison"
C_LL_vincenzo = np.load(f"{path_new}/output/matrici_base/common_ell_and_deltas/Cij_14may/C_LL_nbl30_lmaxWL5000_WLonly.npy")
# C_LG_vincenzo = np.genfromtxt(f"{path_new}/output/matrici_base/common_ell_and_deltas/Cij_14may/CijGL-GR-Flat-eNLA-NA.dat")
C_GG_vincenzo = np.load(f"{path_new}/output/matrici_base/common_ell_and_deltas/Cij_14may/C_GG_nbl30_lmaxGC3000_GConly.npy")





# remember: all ell_values are equal at the moment
nbl = 30
path_masterUnified_forSSCcomparison = "C:/Users/dscio/Documents/Lavoro/Programmi/master_unified_forSSCcomparison"
ell_LL = np.genfromtxt(f"{path_masterUnified_forSSCcomparison}/output/ell_values/ell_WL_ellMaxWL5000_nbl{nbl}.txt")
ell_GG = np.genfromtxt(f"{path_masterUnified_forSSCcomparison}/output/ell_values/ell_GC_ellMaxGC3000_nbl{nbl}.txt")



i = 4
j = 9
# stacked_plots(C_LL_davide, C_LL_vincenzo, i, j)
# stacked_plots(C_LL_davide, C_LL_vincenzo, i, j)



array_1 = Cij_WFdavide_IA_oldBias_nz300_testing
# array_2 = C_LL_vincenzo
array_2 = C_LL_vincenzo


diff = diff_new(array_1, array_2) 

plt.plot(ell_LL, array_1[:,i,j], ".-", label = namestr(array_1, globals()))
plt.plot(ell_LL, array_2[:,i,j], ".-", label = namestr(array_2, globals()))
# plt.plot(ell_values, C_LG_davide[:,i,j], "--", label = "davide")
# plt.plot(ell_values, diff[:,i,j], "-", label = "diff i = %i" %i)

# plt.plot(np.log10(C_LL_vincenzo_raw[:,0]), C_LL_vincenzo_raw[:,1], label = "vincenzo_raw")

plt.yscale("log")
plt.legend()

# plt.matshow(C_LL_davide_WFdavide_IA_newBias_test[0,:,:])
# plt.colorbar()


# array_toInterpolate = C_LG_davide
# reference_array = wig_vincenzo_2[1:,:]

# interpolated_array = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in reference_array[:,0]] )

# difference = np.zeros((reference_array.shape[0], zbins + 1))
# difference[:,0] = reference_array[:,0]
# difference[:,column+1] = (reference_array[:, column + 1] - interpolated_array)/reference_array[:, column + 1] * 100

# plot(wig_davide, wig_vincenzo_2, difference, column)



    
    
    
    
    
    