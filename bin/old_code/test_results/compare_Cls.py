from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
params = {'lines.linewidth' : 2.5,
          'font.size' : 17,
          'axes.labelsize': 'large',
          'axes.titlesize':'large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
plt.rcParams.update(params)

def Get_Cl(filename):
    nzbins = 10
    nb_auto = int(nzbins*(nzbins-1)/2 + nzbins)
    nb_X = nzbins*nzbins
    nb_full = nb_X + nb_auto*2
    hdul = fits.open(filename)
    nell = hdul[2].header['N_ANG']
    ell  = hdul[2].data['ANG'][:nell]
    Cl_GC = np.zeros((nb_auto, nell))
    Cl_WL = np.zeros((nb_auto, nell))
    Cl_X  = np.zeros((nb_X, nell))
    print(nell, nb_X, hdul[3].data.shape)
    
#    X Cl
    for b in range(nb_X):
        for l in range(nell):
            Cl_X[b][l] = hdul[3].data[nell*b + l][3]
    
    #Auto Cl
    for b in range(nb_auto):
        for l in range(nell):
            Cl_GC[b][l] = hdul[4].data[nell*b + l][3]
            Cl_WL[b][l] = hdul[2].data[nell*b + l][3]
    
    hdul.close()
    
    return [Cl_WL, Cl_GC, Cl_X]

def interpolate_vincenzo(nbl, ell_values):    
    vincenzo_interp = np.zeros((nbl, npairs))
    for i in range(npairs):
        f = interp1d(vincenzo_original[:,0], vincenzo_original[:,i+1], kind = "linear")
        vincenzo_interp[:,i] = f(ell_values) 
    return vincenzo_interp
    
def matshow(array):
    plt.matshow(array)
    plt.colorbar()
    plt.title("diff")


def reshape(array, npairs): # reshape Cij from 3D to 2D
    ind = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/indici.dat")
    ind = ind.astype(int)
    ind = ind - 1 
    
    nbl = array.shape[0]
    
    # setting the ell bounds    
    ind_LL = ind[:55, 2:] # the ij indices for WL and GC
    ind_LG = ind[55:155, 2:] # the ij indices for XC

    print(npairs, nbl) #debug
    output_2D = np.zeros((nbl, npairs)) # creating output 2- dimensional array
    
    ind_probe = ind_LL # debug
    
    if   (npairs == 55):  ind_probe = ind_LL
    elif (npairs == 100): ind_probe = ind_LG

    # filling the output array
    for ell in range(nbl):
        for i in range(npairs):
            output_2D[ell, i] =  array[ell, ind_probe[i, 0], ind_probe[i, 1]]
    
    return output_2D
    
def get_kv_pairs(folder):
    from pathlib import Path
    for path in Path(folder).glob("*.txt"):
        yield path.stem, np.genfromtxt(str(path))
    
###############################################################################

path     = "C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison"
path_Cij = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output"

ind = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/indici.dat")
ind = ind.astype(int)
ind = ind - 1 #xxx attenzione!!!

ind_LL = ind[:55,2:] # the ij indices for WL
ind_LG = ind[55:155,2:] # the ij indices for XC
ind_GG = ind[155:,2:] # the ij indices for GC (= ij indices for WL)

ell_min = 10
ell_max = 5000
nbl     = 40

ell_nbl40 = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl)  # to reproduce sylvain's - NOT nbl+1!
ell_nbl20 = np.genfromtxt("%s/ell_values/ell_WL.dat" %path_Cij) # these are in log scale


sylvain_filename = "%s/data/sylvain/fits_sylvain/euclid_simulation_GCph_WL_XC_5000_fsky0.375_gaussian_alone.fits" %path
Cl_WL, Cl_GC, Cl_X = Get_Cl(sylvain_filename)


folder_Cijdavide = "Cijs_v19_ALL/Cij_WFdavide_IA_oldBias_nz300_testing"
folder_Cijdavide = "Cijs_v19_ALL/Cij_WFdavide_IA_newBias_nz300"
folder_Cijdavide = "Cijs_v19_ALL/Cij_WFvincenzo_noIA_oldBias_nz300"
folder_Cijdavide = "Cijs_v19_ALL/Cij_WFsylvain_noIA_oldBias_nz7000"


selector = "LL"

if selector == "LL":
    sylvain           = Cl_WL
    davide            = np.genfromtxt("%s/output/matrici_base/CijLL2x2_prova_sylvain.dat" %path)
    davide_WFdavide   = np.genfromtxt("%s/Cij/%s/C_LL_2D.txt" %(path_Cij, folder_Cijdavide)) # if txt
    # davide_WFdavide   = np.load("%s/Cij/%s/Cij_LL.npy" %(path_Cij, folder_Cijdavide)) # if npy
    vincenzo_original = np.genfromtxt("%s/data/vincenzo/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat" %path)
    npairs = 55
elif selector == "LG":
    sylvain           = Cl_X
    davide            = np.genfromtxt("%s/output/matrici_base/CijLG2x2_prova_sylvain.dat" %path)
    davide_WFdavide   = np.genfromtxt("%s/Cij/%s/C_LG_2D.txt" %(path_Cij, folder_Cijdavide)) # if txt
    # davide_WFdavide   = np.load("%s/Cij/%s/Cij_LG.npy" %(path_Cij, folder_Cijdavide)) # if npy
    vincenzo_original = np.genfromtxt("%s/data/vincenzo/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat" %path)
    npairs = 100
elif selector == "GG":
    sylvain           = Cl_GC
    davide            = np.genfromtxt("%s/output/matrici_base/CijGG2x2_prova_sylvain.dat" %path)
    davide_WFdavide   = np.genfromtxt("%s/Cij/%s/C_GG_2D.txt" %(path_Cij, folder_Cijdavide)) # if txt
    # davide_WFdavide   = np.load("%s/Cij/%s/Cij_LG.npy" %(path_Cij, folder_Cijdavide)) # if npy
    vincenzo_original = np.genfromtxt("%s/data/vincenzo/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat" %path)
    npairs = 55

# this is to load them in a dictionary
# folder = "%s/Cij/%s" %(path_Cij, folder_Cijdavide)
# davide_WFdavide_dict = dict(get_kv_pairs(folder))

# prepare matrices for comparison
sylvain = np.transpose(sylvain)
vincenzo_interp = interpolate_vincenzo(nbl = 20, ell_values = 10**ell_nbl20)

# davide_WFdavide = reshape(davide_WFdavide, npairs) # needed for 3dim npy matrices
# np.savetxt("%s/Cij/%s/C_%s.txt" %(path_Cij, folder_Cijdavide, selector), davide_WFdavide) # save the 2dim matrix 
# in txt for convenience

# saving the interpolated values
# np.savetxt("%s/data/Cij-NonLin-eNLA_15gen/interpolations_4mar/C_LG_interp_nbl40.dat" %path, vincenzo_interp)

diff = (vincenzo_interp-davide_WFdavide)/vincenzo_interp*100
diff = np.abs(diff)

row = 10
# for row in range(npairs):
plt.plot(10**ell_nbl20, np.abs(davide_WFdavide[:, row]), "--", label = "abs(davide_WFdavide %s)" %folder_Cijdavide)
plt.plot(10**ell_nbl20, np.abs(vincenzo_interp[:, row]), ".-", label = "abs(vincenzo_interp)")
plt.plot(ell_nbl40, np.abs(sylvain[:, row]), "--", label = "abs(sylvain)")
# plt.plot(ell_nbl40, np.abs(davide[:, row]), "--", label = "abs(davide)")


plt.yscale("log")
plt.xscale("log")
plt.xlabel("$\ell$")
plt.title("$C^{%s}_{ij}(\ell)$" %selector)
plt.legend()
plt.grid()
plt.show()

# plt.plot(range(npairs), diff[0, :], ".-", label = "diff")


########## matshow ###############
# matshow(diff)

############ tests ###############
# MAX = np.amax(diff)
# position = np.where(diff == MAX)
# print("Maximum % diff and its position: ", MAX, position[0], position[1])

# average = np.average(diff)
# print("average percent difference: %i%%" %average)

################# boh, leftovers ############
# upper_lim = sylvain.shape[1]
# x = np.linspace(0, upper_lim-1, upper_lim)

# for i in range(0, sylvain.shape[0], 7):
#     plt.plot(x, np.abs(sylvain[i,:]), "r.-", label = "Sylvain")
#     plt.plot(x, np.abs(davide[i,:]),  "b.-", label = "Davide")
#     plt.yscale("log")
#     plt.xlabel("z bin index")
#     plt.ylabel("Cij: blue = Davide, red = Sylvain")
#     plt.title("Cl_GC")


#selector = "ell"
#
#if (selector == "z"):
#  
#    upper_lim = sylvain.shape[1]
#    x = np.linspace(0, upper_lim-1, upper_lim)
#    
#    for i in range(0,diff.shape[0]):
#        plt.plot(x, diff[i,:],  label="%i" %i)
#        plt.ylabel("(Cl_Davide - Cl_Sylvain) / Cl_Davide * 100")
#        plt.xlabel("z bin index")
#    #    plt.yscale("log")
#    #    plt.legend()
#        plt.title("% difference - WL")
#        
#elif(selector == "ell"):
#  
#    upper_lim = sylvain.shape[0]
#    x = np.linspace(0, upper_lim-1, upper_lim)
#    
#    for i in range(0,diff.shape[1]):
#        plt.plot(x, diff[:,i],  label="%i" %i)
#        plt.ylabel("(Cl_Davide - Cl_Sylvain) / Cl_Davide * 100")
#        plt.xlabel("ell bin index")
#    #    plt.yscale("log")
#    #    plt.legend()
#        plt.title("% difference - WL")



    
    
