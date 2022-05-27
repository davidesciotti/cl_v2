import numpy as np
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from numba import jit

#from functools import partial
#from astropy.cosmology import WMAP9 as cosmo
#from astropy import constants as const
#from astropy import units as u

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"
start = time.time()

# obiettivo: formula 103 paper IST

c = 299792.458 # km/s is the unit correct??
H0 = 67 #km/(s*Mpc)

Om0  = 0.32
Ode0 = 0.68
Ox0  = 0
gamma = 0.55

z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus  = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))

z_mean  = (z_plus + z_minus)/2
z_min   = z_minus[0]
z_max   = z_plus[9]
z_max   = 4

f_out = 0.1
sigma_b = 0.05
sigma_o = 0.05
c_b = 1.0
c_o = 1.0
z_b = 0
z_o = 0.1

z_m = 0.9
z_0 = z_m/np.sqrt(2)

A_IA = 1.72
C_IA = 0.0134
eta_IA = -0.41
beta_IA = 2.17

zbins = 10

####################################### function definition
def E(z):
    result = np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result

def inv_E(z):
    result = 1/np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result
    
def r_tilde(z):
    #r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    result = quad(inv_E, 0, z)  # integrate 1/E(z) from 0 to z
    return result[0]

def r(z):
    result = c/H0 * quad(inv_E, 0, z)[0] 
    return result
    
def pph(z, z_p):    
    first_addendum  = (1-f_out)/(np.sqrt(2*np.pi)*sigma_b*(1+z)) * \
                      np.exp(-0.5 * ( (z-c_b*z_p-z_b)/(sigma_b*(1+z)) )**2)
    second_addendum = (f_out)/(np.sqrt(2*np.pi)*sigma_o*(1+z)) * \
                      np.exp(-0.5 * ( (z-c_o*z_p-z_o)/(sigma_o*(1+z)) )**2)      
    return first_addendum + second_addendum

def n(z):
    result = (z/z_0)**2 * np.exp(-(z/z_0)**(3/2))
    result = result*(30/0.4242640687118783) # normalising the distribution ? xxx
    return result

############ section to delete: checking if ni(z) is the issue
n_i_import = np.load(r"%s\output\base_functions_v0\n_i(z,i).npy" %path)
def n_i(z,i):
    n_i_interp = interp1d(n_i_import[:,0], n_i_import[:,i+1], kind = "linear")
    result_array = n_i_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

def wil_tilde(z, i): #xxx attention, check carefully
    integrand = lambda z_prime, z, i: n_i(z_prime, i)*(1 - r_tilde(z)/r_tilde(z_prime))
    # integrate in z_prime, it must be the first argument
    result = quad(integrand, z, z_max, args = (z, i), limit = 100)
    return result[0]

def wil(z,i):
    return (3/2)*(H0/c)*Om0*(1+z)*r_tilde(z)*wil_tilde(z,i)

########################################################### IA

def W_IA(z,i):
    result = (H0/c)*n_i(z,i)*E(z)
    return result

lumin_ratio = np.genfromtxt("%s\data\scaledmeanlum-E2Sa_EXTRAPOLATED.txt" %path)
def L_ratio(z):
    lumin_ratio_interp1d = interp1d(lumin_ratio[:,0], lumin_ratio[:,1], kind='linear')
    result_array = lumin_ratio_interp1d(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array OK!
    return result

def F_IA(z):
    result = (1+z)**eta_IA * (L_ratio(z))**beta_IA
    return result

# use formula 23 for Om(z)
def Om(z):
    return Om0*(1+z)**3 / E(z)**2

def D(z):
    integrand = lambda x: Om(x)**gamma / (1+x)
    integral = quad(integrand, 0, z)[0]
    result = np.exp(-integral)
    return result

def wil_tot(z,i):
    # xxx Om0 o Om(z)? formula 135
    return wil(z,i) - (A_IA*C_IA*Om0*F_IA(z))/D(z) * W_IA(z,i)

###################### wig ###########################
def b(i):
    return np.sqrt(1+z_mean[i])

# n_bar = np.genfromtxt(r"%s\output\base_functions_v0\n_bar.txt" %path)
n_bar = np.genfromtxt(r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide\output\base_functions_v0\n_bar.txt")
# optimization: I don't need this anymore
# def n_bar_i(i):
#     result = quad(n_i, z_min, z_max, args = i, limit=100 ) 
#     return result[0]


def wig(z,i):
    return b(i)* (n_i(z,i)/n_bar[i]) *H0*E(z)/c
    # xxx I'm already dividing by c!

########################### modify WFs according to Overleaf draft
def wil_SSC(z,i):
    return wil_tot(z,i)/r(z)**2
    
def wig_SSC(z,i):
    return wig(z,i)/r(z)**2
###################################################






def compute_1D(function, z_array_toInsert):
    array = np.zeros((z.shape[0], 2)) # it has only one column apart from z
    array[:,0] = z_array_toInsert
    array[:,1] = np.asarray([function(zi) for zi in z])
    return array

def compute_2D(function, z_array_toInsert):

    array = np.zeros((z.shape[0], zbins+1))
    array[:,0] = z_array_toInsert
    
    for i in range(zbins):
        array[:,i+1] = np.asarray([function(zi, i) for zi in z])  
        
        print("debug: I'm working on column number %i" %i)
        end = time.time()
        print("the program took %i seconds to run" %(end - start))
    return array


def save(array, name):
    np.save(r"%s\output\base_functions_v0\%s.npy" %(path, name), array)
    
    
########################### computing and saving the functions
zsteps = 300
z = np.linspace(z_min, z_max, zsteps)

array_appoggio = compute_2D(wig_SSC, z)
save(array_appoggio, "wig_SSC(z,i)")
print("wig_SSC DONE")
array_appoggio = compute_2D(wig, z)
save(array_appoggio, "wig(z,i)")
print("wig DONE")

array_appoggio = compute_2D(wil_tilde, z)
save(array_appoggio, "wil_tilde(z,i)")
print("wil_tilde DONE")
array_appoggio = compute_2D(wil, z)
save(array_appoggio, "wil(z,i)")
print("WIL DONE")
array_appoggio = compute_2D(wil_tot, z)
save(array_appoggio, "wil_tot(z,i)")
print("wil_tot DONE")
array_appoggio = compute_2D(wil_SSC, z)
save(array_appoggio, "wil_SSC(z,i)")
print("wiL_SSC DONE")
array_appoggio = compute_1D(L_ratio, z)
save(array_appoggio, "L_ratio(z,i)")
print("L_ratio DONE")

end = time.time()
print("the program took %i seconds to run" %(end - start))

#r_check = np.load("%s/output/base_functions/inv_E(z).npy" %(path))

# optimization: saving n_bar once and for all
# n_bar_toSave = np.zeros((zbins))
# for i in range(zbins):
#     n_bar_toSave[i] = n_bar_i(i)
# np.savetxt(r"%s\output\base_functions_v0\n_bar.txt" %path, n_bar_toSave)

