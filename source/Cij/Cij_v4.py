import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

start = time.time()


c  = 299792.458 # km/s is the unit correct??
H0 = 67         # km/(s*Mpc)

Om0  = 0.32
Ode0 = 0.68
Ox0  = 0
gamma = 0.55

z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus  = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))
z_mean  = (z_plus + z_minus)/2
z_min   = z_minus[0]
z_max   = z_plus[9]

f_out = 0.1
sigma_b = 0.05
sigma_o = 0.05
c_b = 1.0
c_o = 1.0
z_b = 0
z_o = 0.1

z_m = 0.9
z_0 = z_m/np.sqrt(2)

zbins = 10
nbl   = 20

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
    # attention: quad(...) returns a tuble. I can't multiply a float (c/H0) 
    # with a tuple.
    return result

def k_limber(ell,z):
    return (ell+1/2)/r(z)

# this function is to interpolate (2-dimensional) the PS previously generated
PS = np.load(r"%s\output\P(k,z).npy" %path)
PS_transposed = PS.transpose()

h = 0.67
k_max =50/h # xxx unità di misura? * o / h? kmax??
z_array = np.linspace(0.000001, 3,   num=303) # not ok

zmin = 0.01
zmax = 4
zsteps = 200
z_array = np.linspace(zmin, zmax,   zsteps)

k_array = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)

# I think the argument names are unimportant, I could have called it k, k_ell
# or k_limber xxx
def P(k_ell,z):
    # order: interp2d(k, z, PS, kind = "linear") first x, then y
    # is is necessary to transpose, the PS array must have dimensions
    # [z.ndim, k.ndim]. I have it the other way round. The console says:
    # ValueError: When on a regular grid with x.size = m and y.size = n, 
    # if z.ndim == 2, then z must have shape (n, m)
    PS_interp = interp2d(k_array, z_array, PS_transposed[:,:], kind = "linear")
    result_array = PS_interp(k_ell,z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array OK!
    return result

wil_import = np.load(r"%s\output\base_functions_v1\wil_tot(z,i).npy" %path)
def wil_tot(z,i):
    # it's i+1: first column is for the redshift array
    wil_interp = interp1d(wil_import[:,0], wil_import[:,i+1], kind = "linear")
    result_array = wil_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

wig_import = np.load(r"%s\output\base_functions_v1\wig(z,i).npy" %path)
def wig(z,i):
    # it's i+1: first column is for the redshift array
    wig_interp = interp1d(wig_import[:,0], wig_import[:,i+1], kind = "linear")
    result_array = wig_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

################## Cijs
def Cij_LL(i,j,ell): 
    def integrand(z, i,j,ell): # first argument is the integration variable
        return wil_tot(z,i)*wil_tot(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
    # check z_min, z_max xxx
    return result

def Cij_GL(i,j,ell): # xxx GL or LG?
    def integrand(z, i,j,ell): # first argument is the integration variable
        return wig(z,i)*wil_tot(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell))[0]
    return result

def Cij_GG(i,j,ell):
    def integrand(z, i,j,ell): # first argument is the integration variable
        return wig(z,i)*wig(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
    result = c/H0 * quad(integrand, z_min, z_max, args = (i,j,ell), limit = 100,  full_output=True)[0]
    # xxx maybe you can try with scipy.integrate.romberg
    return result
######################
lmin = 10
lmax = 3000
ell_values = np.logspace(np.log10(lmin), np.log10(lmax), nbl)
# attento ai valori di ell!!

def compute():
    C_LL = np.zeros((zbins, zbins, nbl))
    C_GL = np.zeros((zbins, zbins, nbl))
    C_GG = np.zeros((zbins, zbins, nbl)) 

    for i in range(zbins):
        for j in range(zbins):
            k = 0
            print("i = %i, j = %i" %(i,j))
            for ell in ell_values:
                print("k = %i" %k)
                # C_LL[i,j,k] = Cij_LL(i,j,ell)
                # C_GL[i,j,k] = Cij_GL(i,j,ell)
                C_GG[i,j,k] = Cij_GG(i,j,ell)
                k = k+1
    
    return(Cij_LL, Cij_GL, Cij_GG)

def save():
    np.save(r"%s\output\Cijs\Cij_LL.npy" %path, C_LL)
    np.save(r"%s\output\Cijs\Cij_LG.npy" %path, C_LG)
    np.save(r"%s\output\Cijs\Cij_GG.npy" %path, C_GG)
################################################### end of function declaration

C_LL, C_LG, C_GG = compute()
save()

# xxx problem: the integrand is very large (O(10^2))
#def analyze_integrand(z,i,j,ell):
#    return wig(z,i)*wig(z,j) / (E(z)*r(z)**2) * P(k_limber(ell,z), z)
#
#z = np.linspace(0.01, 0.6, 500)
#vals = np.asarray([analyze_integrand(zi, 0, 0, 0) for zi in z])
#plt.plot(z, vals, "b.-")


#plt.plot(z_array, PS[row, :], label = "PS originale, z = %f" %z_array[column])
#plt.plot(z_array_2, vals, label = "P(k,z) interpolato, z = %f" %z_array[column])
#plt.ylabel("P(k, z)")
#plt.title("Matter Power Spectrum")
#plt.legend()
#plt.grid("true", linestyle = "--")
##    plt.yscale("log")
#plt.xlabel("$k$")


end = time.time()
print("the program took %i seconds to run" %(end - start))
