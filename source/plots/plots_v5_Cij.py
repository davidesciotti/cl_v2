import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16


path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

def interpolation(z, array_toInterpolate, column):
    interpolation = interp1d(array_toInterpolate[:,0], array_toInterpolate[:,column + 1], kind = "linear")
    result_array = interpolation(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

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
    axes[0].set(xlabel='$\ell$', ylabel= "$\dfrac{|W_i^{L, davide}(z) - W_i^{L, vincenzo}(z)|}{W_i^{L, davide}(z)}$", title = '$\Delta W_i^L(z)/W_i^{L, davide}$ (percent), i = %i' %column)
    # axes[0].legend(prop={'size': 14})
    axes[0].grid()
    
    axes[1].plot(ell_values, array_davide[:,i,j], ".-", label = "davide") 
    axes[1].plot(ell_values, array_vincenzo[:,i,j], ".-", label = "vincenzo")
    axes[1].set(xlabel='$\ell$', ylabel='$W_i^L(z)$', title = "$W_i^L(z)$, i = %i" %column)
    axes[1].legend(prop={'size': 14})
    axes[1].grid()
    
    axes[2].plot(ell_values, array_davide[:,i,j], ".-", label = "davide") 
    axes[2].plot(ell_values, array_vincenzo[:,i,j], ".-", label = "vincenzo")
    axes[2].set(yscale = "log", xlabel='$\ell$', ylabel='$W_i^L(z)$', title = "$W_i^L(z)$, i = %i" %column)
    axes[2].legend(prop={'size': 14})
    axes[2].grid()
        
    fig.tight_layout()


####################################### function definition
c = 299792.458 # km/s 

zbins = 10
zmin = 0.001
zmax = 4
zsteps = 3000
z = np.linspace(zmin, zmax, zsteps)

# davide ##########################################
Cij_folder = "Cijs_v10_"
C_LL_davide = np.load(r"%s\output\%s\Cij_LL.npy" %(path, Cij_folder))
C_LG_davide = np.load(r"%s\output\%s\Cij_LG.npy" %(path, Cij_folder))
C_GG_davide = np.load(r"%s\output\%s\Cij_GG.npy" %(path, Cij_folder))

# vincenzo ##############################
C_LL_vincenzo_raw = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijLL-LCDM-NonLin-eNLA.dat" %path)
C_LG_vincenzo_raw = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijLG-LCDM-NonLin-eNLA.dat" %path)
C_GG_vincenzo_raw = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijGG-LCDM-NonLin-eNLA.dat" %path)

# anzi, prendo quelli reshaped
C_LL_vincenzo = np.load(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_LL_nbl20.npy" %path)
C_LG_vincenzo = np.load(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_LG_nbl20.npy" %path)
C_GG_vincenzo = np.load(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_GG_nbl20.npy" %path)

# remember: all ell_values are equal at the moment
ell_values = np.genfromtxt(r"%s\output\ell_values\ell_WL.dat" %path)

C_LL_davide_reshape = np.zeros((20,10,10))
C_LG_davide_reshape = np.zeros((20,10,10))
C_GG_davide_reshape = np.zeros((20,10,10))

for i in range(20):
    C_LL_davide_reshape[i,:,:] = C_LL_davide[:,:,i]
    C_LG_davide_reshape[i,:,:] = C_LG_davide[:,:,i]
    C_GG_davide_reshape[i,:,:] = C_GG_davide[:,:,i]


############ yyy section to delete
def load(folder_name):
    C_LL = np.load(r"%s\output\%s\Cij_LL.npy" %(path, folder_name) )
    return C_LL
    
i = j = 0
# C_LL = load("Cijs_v10_WFvincenzo")
# plt.plot(ell_values, C_LL[:,i,j], label = "WF di Vincenzo")
# C_LL = load("Cijs_v4_noIA")
# plt.plot(ell_values, C_LL[:,i,j], label = "no IA")
# C_LL = load("Cijs_v5_noIA_zmax2.5")
# plt.plot(ell_values, C_LL[:,i,j], label = "no IA, zmax = 2.5")
# C_LL = load("Cijs_v9_IA_zmax4")
# plt.plot(ell_values, C_LL[:,i,j], label = "IA, zmax = 4")
# C_LL = load("Cijs_v3")
# plt.plot(ell_values, C_LL[:,i,j], label = "v3")
# C_LL = load("Cijs_v2")
# plt.plot(ell_values, C_LL[:,i,j], label = "v2")
C_LL = load("Cijs_v11_newell")
plt.plot(10**ell_values, C_LL[:,i,j], label = "Cijs_v11_newell")

# C_LL = load("Cijs_v2_recheck")
# plt.plot(ell_values, C_LL[:,i,j], label = "v2 recheck (non cambia nulla)")
# C_LL = load("Cijs_v2_recheck\correct_ordering")
# plt.plot(ell_values, C_LL[:,i,j], label = "v2 recheck (non cambia nulla)")
# C_LL = np.load(r"%s\output\Cij_LL.npy" %path)
# plt.plot(ell_values, C_LL[:,i,j], label = "v2 recheck (non cambia nulla)")


plt.plot(C_LL_vincenzo_raw[:,0], C_LL_vincenzo_raw[:,1], label = "vincenzo raw")

# plt.plot(ell_values, C_LL_vincenzo[:,i,j], label = "vincenzo original")

# plt.plot(ell_values, C_LL_davide_reshape[:,i,j], label = "davide nbl_last (=v2?)")
plt.legend()


############ yyy end section to delete


# i = 0
# j = 0
# stacked_plots(C_LL_davide, C_LL_vincenzo, i, j)



# plot(C_GG_davide, C_GG_vincenzo, i, j)
# diff = compare(C_LL_davide, C_LL_vincenzo, i, j)

# plt.plot(ell_values, C_LL_davide[:,0,0], ".-", label = "davide")
# plt.plot(ell_values, C_LL_davide[:,i,j], ".-", label = "davide")
# plt.plot(ell_values, C_LL_vincenzo[:,i,j], ".-", label = "vincenzo")
# plt.plot(np.log10(C_LL_vincenzo_raw[:,0]), C_LL_vincenzo_raw[:,1], label = "vincenzo_raw")
# plt.plot(ell_values, diff, ".-", label = "diff")

# plt.yscale("log")
# plt.legend()



# array_toInterpolate = C_LG_davide
# reference_array = wig_vincenzo_2[1:,:]

# interpolated_array = np.asarray( [interpolation(zi, array_toInterpolate, column) for zi in reference_array[:,0]] )

# difference = np.zeros((reference_array.shape[0], zbins + 1))
# difference[:,0] = reference_array[:,0]
# difference[:,column+1] = (reference_array[:, column + 1] - interpolated_array)/reference_array[:, column + 1] * 100

# plot(wig_davide, wig_vincenzo_2, difference, column)



    
    
    
    
    
    