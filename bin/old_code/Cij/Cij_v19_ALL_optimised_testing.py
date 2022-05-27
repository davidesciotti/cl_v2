import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time
from numba import jit

path = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide"
path_SSC = "C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison"

start = time.time()

c  = 299792.458 # km/s
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

zbins = 10
nbl   = 20

h = 0.67
k_max = 30 # xxx unità di misura? * o / h? kmax??

# zsteps = 200
# z_array = np.linspace(z_min, z_max,   zsteps)
z_array = np.linspace(0.000001, 4,   num=401) # not ok
z_array = np.linspace(0.000001, 3,   num=303) # not ok zzzzzzzzzzzzzzzz
k_array = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)
# TODO fix z_array -> fatto?

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
PS = np.genfromtxt("%s/data/Power_Spectrum/Pnl.txt" %path) # it is the non linear one!!
PS = np.load(r"%s\output\P(k,z).npy" %path) # zzzzzzzzzzzzzzz
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



for i in range(1):
    ############################ DEFINING THE INPUTS ##########################
    # if i == 0: # davide nz300 IA newBias (so import wig noBias) 2.2 hours (7900 sec) to run
    #     selector = "oldBias"
    #     Cij_output_folder = "Cijs_v19_ALL/Cij_WFdavide_IA_oldBias_nz300_testing" #this needs to be fixed
    #     z_max   = 4 # that's the domain of vincenzo's WFs
    #     wil_import = np.genfromtxt("%s/data/everyones_WFs/WF_REORGANIZED_MASTERFOLDER_v2/davide/nz300/wil_davide_IA_IST_nz300.txt" %(path_SSC))
    #     wig_import = np.genfromtxt("%s/data/everyones_WFs/WF_REORGANIZED_MASTERFOLDER_v2/davide/nz300/wig_davide_IST_nz300.txt" %(path_SSC))
    if i == 0: # Sylvain nz7000 noIA
        selector = "newBias"
        Cij_output_folder = "Cijs_v19_ALL/Cij_WFsylvain_noIA_oldBias_nz7000"
        z_max   = 3.7 
        wil_import = np.genfromtxt("%s/data/everyones_WFs/WF_REORGANIZED_MASTERFOLDER_v2/sylvain/wil_sylvain_noIA_IST_nz7000.txt" %(path_SSC))
        wig_import = np.genfromtxt("%s/data/everyones_WFs/WF_REORGANIZED_MASTERFOLDER_v2/sylvain/wig_sylvain_IST_nz7000.txt"    %(path_SSC))
    
    # IA/noIA, old/new/multibinBias are decided in the import section at the beginning of the code
    def wil(z,i):
        # it's i+1: first column is for the redshift array
        wil_interp = interp1d(wil_import[:,0], wil_import[:,i+1], kind = "linear")
        result_array = wil_interp(z) #z is considered as an array
        result = result_array.item() #otherwise it would be a 0d array 
        return result
    
    def wig(z,i):
        # it's i+1: first column is for the redshift array
        wig_interp = interp1d(wig_import[:,0], wig_import[:,i+1], kind = "linear")
        result_array = wig_interp(z) #z is considered as an array
        result = result_array.item() #otherwise it would be a 0d array 
        return result

    
    ################## Cijs
    
    ###### NEW BIAS ##################
    # bias
    b = np.zeros((zbins))
    for i in range(zbins):
        b[i] = np.sqrt(1+z_mean[i])
    
    # integrand
    def K_ij_XC(z,i,j):
        return wil(z,j) * wig(z,i) / (E(z)*r(z)**2)
    def K_ij_GG(z,i,j):
        return wig(z,j) * wig(z,i) / (E(z)*r(z)**2)
        
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
    
    
    ###### OLD BIAS ##################
    def Cij_LL_function(i,j,ell): 
        def integrand(z, i,j,ell): # first argument is the integration variable
            return wil(z,i)*wil(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
        result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
        # check z_min, z_max xxx
        return result
    
    def Cij_LG_function(i,j,ell): # xxx GL or LG? OLD BIAS
        def integrand(z, i,j,ell): # first argument is the integration variable
            return wil(z,j)*wig(z,i) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
        result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
        return result
    
    def Cij_GG_function(i,j,ell): # OLD BIAS
        def integrand(z, i,j,ell): # first argument is the integration variable
            return wig(z,i)*wig(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
        result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell),  full_output=True)[0]
        # xxx maybe you can try with scipy.integrate.romberg
        return result
    
    
    ########### computation functions
    def reshape(array, npairs, name):
        ind = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/indici.dat")
        ind = ind.astype(int)
        ind = ind - 1 #xxx attenzione!!!
        
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
        
        # saving it
        np.savetxt("%s/output/Cij/%s/%s.txt" %(path, Cij_output_folder, name), output_2D)
    
    
    def compute():
        C_LL_array = np.zeros((nbl, zbins, zbins))
        C_LG_array = np.zeros((nbl, zbins, zbins))
        C_GG_array = np.zeros((nbl, zbins, zbins))
    
        k = 0
        for ell in ell_values:
            if ell > ell_values[0]: # the first time k should remain 0 
                k = k+1 
            print("k = %i, ell = %f" %(k, ell))
            end = time.time()
            print("the program took %i seconds to run" %(end - start))
            for i in range(zbins):
                for j in range(zbins):
                    
                    if j >= i: # C_LL and C_GG are symmetric!
                        C_LL_array[k,i,j] = Cij_LL_function(i,j,ell)
                        # if (selector == "oldBias"):
                        #     C_GG_array[k,i,j] = Cij_GG_function(i,j,ell) # old bias
                        # elif (selector == "newBias"):
                        #     C_GG_array[k,i,j] = sum_Cij_GG(i,j,ell) # new bias

                    # if (selector == "oldBias"):
                    #     C_LG_array[k,i,j] = Cij_LG_function(i,j,ell) # old bias
                    # elif (selector == "newBias"):
                    #     C_LG_array[k,i,j] = sum_Cij_LG(i,j,ell) # new bias
                    
                    # finished computing this j value
                    # print("k = %i, i = %i, j = %i, ell = %f" %(k, i,j, ell))      
                    # end = time.time()
                    # print("the program took %i seconds to run" %(end - start))
    
        return(C_LL_array, C_LG_array, C_GG_array)
    
    
    def fill_symmetric_Cls():
        for k in range(ell_values.shape[0]):
            for i in range(zbins):
                for j in range(zbins):
                    if j < i: # C_LL and C_GG are symmetric!
                        C_LL_array[k,i,j] = C_LL_array[k,j,i]
                        C_GG_array[k,i,j] = C_LL_array[k,j,i]
    
    
    def save():
        np.save("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_output_folder), C_LL_array)
        np.save("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_output_folder), C_LG_array)
        np.save("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_output_folder), C_GG_array)
        
    ################################################### end of function declaration
        
    
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
    C_LL_array, C_LG_array, C_GG_array = compute()
    fill_symmetric_Cls()
    save()
    
    ############### reshape to compare with others ##########
    reshape(C_LL_array, 55,  "C_LL_2D")
    reshape(C_LG_array, 100, "C_LG_2D")
    reshape(C_GG_array, 55,  "C_GG_2D")
    
    
    end = time.time()
    print("the program took %i seconds to run" %(end - start))
