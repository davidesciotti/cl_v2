import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time
from numba import jit

path = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide"
path_SSC = "C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison"

start = time.time()

c  = 299792.458 # km/s is the unit correct??
H0 = 67         # km/(s*Mpc)
Om0  = 0.32
Ode0 = 0.68
Ox0  = 0

z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus  = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))

z_mean  = (z_plus + z_minus)/2
z_min   = z_minus[0]
z_max   = z_plus[9]
# xxx is z_max = 4 to be used everywhere?
z_max   = 4
z_max   = 3.73 # that's the domain of vincenzo's WFs

zbins = 10
nbl   = 20

h = 0.67
k_max = 30 # xxx unità di misura? * o / h? kmax??

# zsteps = 200
# z_array = np.linspace(z_min, z_max,   zsteps)
z_array = np.linspace(0.000001, 3,   num=303) # not ok 
k_array = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)
# TODO fix z_array

npairs = 55

WF_input_folder = "everyones_latest"
Cij_output_folder = "Cijs_v17_newBias_WFsylvain"

################### defining the functions
def E(z):
    result = np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result

def inv_E(z):
    result = 1/np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result
    
def r(z):
    #r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    result = c/H0 * quad(inv_E, 0, z)[0]  # integrate 1/E(z) from 0 to z
    return result

def k_limber(ell,z):
    return (ell+1/2)/r(z)

# I think the argument names are unimportant, I could have called it k, k_ell
# or k_limber xxx
# this function is to interpolate (2-dimensional) the PS previously generated
PS = np.load("%s/output/P(k,z).npy" %path) # xxx must be the non linear one!!
PS_transposed = PS.transpose()
def P(k_ell,z):
    # order: interp2d(k, z, PS, kind = "linear") first x, then y
    # is is necessary to transpose, the PS array must have dimensions
    # [z.ndim, k.ndim]. I have it the other way round. The console says:
    # ValueError: When on a regular grid with x.size = m and y.size = n, 
    # if z.ndim == 2, then z must have shape (n, m)
    PS_interp = interp2d(k_array, z_array, PS_transposed, kind = "linear")
    result_array = PS_interp(k_ell, z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array OK!
    return result

# wil_import = np.genfromtxt("%s/data/everyones_WFs/%s/wil_sylvain_noSSC.txt" %(path_SSC, WF_input_folder)) # careful, these WF don't have IA
# def wil_tot(z,i):
#     # it's i+1: first column is for the redshift array
#     wil_interp = interp1d(wil_import[:,0], wil_import[:,i+1], kind = "linear")
#     result_array = wil_interp(z) #z is considered as an array
#     result = result_array.item() #otherwise it would be a 0d array 
#     return result
wil_import = np.genfromtxt("%s/data/everyones_WFs/%s/wil_sylvain_noSSC.txt" %(path_SSC, WF_input_folder))
def wil(z,i):
    # it's i+1: first column is for the redshift array
    wil_interp = interp1d(wil_import[:,0], wil_import[:,i+1], kind = "linear")
    result_array = wil_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

# here wig -> wig_nobias
wig_import = np.genfromtxt("%s/data/everyones_WFs/%s/wig_sylvain_noBias_noSSC.txt" %(path_SSC, WF_input_folder))
def wig_noBias(z,i):
    # it's i+1: first column is for the redshift array
    wig_interp = interp1d(wig_import[:,0], wig_import[:,i+1], kind = "linear")
    result_array = wig_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

################## Cijs
# bias
b = np.zeros((zbins))
for i in range(zbins):
    b[i] = np.sqrt(1+z_mean[i])

# integrand
def K_ij_XC(z,i,j):
    return wil(z,j) * wig_noBias(z,i) / (E(z)*r(z)**2)
def K_ij_GG(z,i,j):
    return wig_noBias(z,j) * wig_noBias(z,i) / (E(z)*r(z)**2)
    
# integral
def Cij_LG_partial(i,j, nbin, ell):
    def integrand(z,i,j, ell):
        return K_ij_XC(z,i,j) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_minus[nbin], z_plus[nbin], args = (i,j,ell))[0]
    return result

def Cij_GG_partial(i,j, nbin, ell):
    def integrand(z,i,j, ell):
        return K_ij_GG(z,i,j) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_minus[nbin], z_plus[nbin], args = (i,j,ell))[0]
    return result

# summing the partial integrals
def sum_Cij_LG(i,j,ell):
    result = 0
    for nbin in range(zbins):
        result += Cij_LG_partial(i,j, nbin, ell)*b[nbin]
    return result

def sum_Cij_GG(i,j,ell):
    result = 0
    for nbin in range(zbins):
        result += Cij_GG_partial(i,j, nbin, ell)*(b[nbin]**2)
    return result



def Cij_LL_function(i,j,ell): 
    def integrand(z, i,j,ell): # first argument is the integration variable
        return wil(z,i)*wil(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
    # check z_min, z_max xxx
    return result

# def Cij_LG_function(i,j,ell): # xxx GL or LG?
#     def integrand(z, i,j,ell): # first argument is the integration variable
#         return wil(z,j)*wig(z,i) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
#     result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
#     return result

# def Cij_GG_function(i,j,ell):
#     def integrand(z, i,j,ell): # first argument is the integration variable
#         return wig(z,i)*wig(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
#     result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell), limit = 100,  full_output=True)[0]
#     # xxx maybe you can try with scipy.integrate.romberg
#     return result

def reshape(array, name):
    ind = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/indici.dat")
    ind = ind.astype(int)
    ind = ind - 1 #xxx attenzione!!!
    
    # setting the ell bounds    
    ind_LL = ind[:55,2:] # the ij indices for WL and GC
    ind_LG = ind[55:155,2:] # the ij indices for XC

    npairs = array.shape[1] # 55 or 100
        
    output_2D = np.zeros((nbl, npairs)) # creating output 2- dimensional array
    
    if (npairs == 55):    ind_probe = ind_LL
    elif (npairs == 100): ind_probe = ind_LG

    # filling the output array
    for ell in range(nbl):
        for i in range(npairs):
            output_2D[ell, i] =  array[ell, ind_probe[i, 0], ind_LL[i, 1]]
    
    # saving it
    np.savetxt("%s/output/Cij/%s/%s.txt" %(path, Cij_output_folder, name), output_2D)
    




######################

# this is now useless. interpolations for the appropriate values will be performed later!
# actually it's better to do it now, to compute 20 values instead of 999
ell_LL = np.genfromtxt("%s/output/ell_values/ell_WL.dat" %path)
# ell_LG = np.genfromtxt("%s/output/ell_values/ell_XC.dat" %path)
# ell_GG = np.genfromtxt("%s/output/ell_values/ell_GC.dat" %path)

# these are Vincenzo's ell values: 
# ell_min = 10
# ell_max = 50000
# ell_steps = 999
# ell_values = np.linspace(ell_min, ell_max, ell_steps)

# XXX aggiunta fondamentale
ell_values = 10**ell_LL


def compute():
    C_LL_array = np.zeros((nbl, zbins, zbins))
    C_LG_array = np.zeros((nbl, zbins, zbins))
    C_GG_array = np.zeros((nbl, zbins, zbins))

    k = 0
    for ell in ell_values:
        if ell > ell_values[0]: # the first time k should remain 0 
            k = k+1 
        print("k = %i" %k)
        for i in range(zbins):
            for j in range(zbins):
                print("k = %i, i = %i, j = %i, ell = %f" %(k, i,j, ell))      
                C_LL_array[k,i,j] = Cij_LL_function(i,j,ell)
                C_LG_array[k,i,j] = sum_Cij_LG(i,j,ell)
                C_GG_array[k,i,j] = sum_Cij_GG(i,j,ell)
    
    return(C_LL_array, C_LG_array, C_GG_array)

def save():
    np.save("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_output_folder), C_LL_array)
    np.save("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_output_folder), C_LG_array)
    np.save("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_output_folder), C_GG_array)
################################################### end of function declaration

C_LL_array, C_LG_array, C_GG_array = compute()
save()

############### reshape to compare with others ##########
reshape(C_LL_array, "C_LL_2D.txt")
reshape(C_LG_array, "C_LG_2D.txt")
reshape(C_GG_array, "C_GG_2D.txt")


end = time.time()
print("the program took %i seconds to run" %(end - start))
