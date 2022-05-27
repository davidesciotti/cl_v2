import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time
from classy import Class

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"

start = time.time()


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

f_out = 0.1
sigma_b = 0.05
sigma_o = 0.05
c_b = 1.0
c_o = 1.0
z_b = 0
z_o = 0.1

z_m = 0.9
z_0 = z_m/np.sqrt(2)

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
    result = c/H0 * quad(inv_E, 0, z)  # integrate 1/E(z) from 0 to z
    return result[0]

def k_limber(ell,z):
    return (ell+1/2)/r(z)

# this function is to interpolate (2-dimensional) the PS previously generated
PS = np.load("%s/output/P(k,z).npy" %path)
PS_transposed = PS.transpose()

h = 0.67
k_max =50/h # xxx unità di misura? * o / h? kmax??
z_array = np.linspace(0.000001, 3,   num=303)
k_array       = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)

def P(k,z):
    # order: interp2d(k, z, PS, kind = "linear") first x, then y
    # is is necessary to transpose, the PS array must have dimensions
    # [z.ndim, k.ndim]. I have it the other way round. The console says:
    # ValueError: When on a regular grid with x.size = m and y.size = n, 
    # if z.ndim == 2, then z must have shape (n, m)
    PS_interp = interp2d(k_array, z_array, PS_transposed[:,:], kind = "linear")
    result_array = PS_interp(k,z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array OK!
    return result


# TODO: risolvi k_limber vs k, nested functions...



#riga = 10
#colonna = 10 
#k_prova = k_array[riga] + k_array[riga]*0.1
#z_prova = z_array[colonna] + z_array[colonna]*0.1

column = 10
row = 300
k_test = 0.5

k_array_2    = np.logspace(np.log10(5e-5), np.log10(30), num=470)
z_array_2 = np.linspace(0.000001, 2.7,   num=100)

vals = np.asarray([P(k_array[row], zi) for zi in z_array_2])




plt.plot(z_array, PS[row, :], label = "PS originale, z = %f" %z_array[column])
plt.plot(z_array_2, vals, label = "P(k,z) interpolato, z = %f" %z_array[column])
plt.ylabel("P(k, z)")
plt.title("Matter Power Spectrum")
plt.legend()
plt.grid("true", linestyle = "--")
#    plt.yscale("log")
plt.xlabel("$k$")


end = time.time()
print("the program took %i seconds to run" %(end - start))
