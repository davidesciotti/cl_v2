import numpy as np
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
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
E_import = np.load(r"%s\output\base_functions\E(z).npy" %path)
def E(z):
    E_interp = interp1d(E_import[:,0], E_import[:,1], kind = "linear")
    result_array = E_interp(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array 
    return result

zsteps = 100
z = np.linspace(0.01, 2.20, zsteps)
array = np.asarray([E(zi) for zi in z])
