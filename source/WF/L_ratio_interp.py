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
lumin_ratio = np.genfromtxt("%s\data\scaledmeanlum-E2Sa.dat" %path)
def L_ratio(z):
    lumin_ratio_interp1d = interp1d(lumin_ratio[:,0], lumin_ratio[:,1], kind='linear')
    result_array = lumin_ratio_interp1d(z) #z is considered as an array
    result = result_array.item() #otherwise it would be a 0d array OK!
    return result

def extrapolate(array, x, selector):

    if selector == "right":
        x1 = array[lumin_ratio.shape[0]-2, 0]    
        y1 = array[lumin_ratio.shape[0]-2, 1]    
        x2 = array[lumin_ratio.shape[0]-1, 0]    
        y2 = array[lumin_ratio.shape[0]-1, 1]   
    elif selector == "left":
        x1 = array[1, 0]    
        y1 = array[1, 1]    
        x2 = array[0, 0]    
        y2 = array[0, 1]   
        
    m = (y2-y1)/(x2-x1)
    q = y1 - m * x1
    y = m*x + q
    return y

    
z_min_L = lumin_ratio[0,0]
z_max_L = lumin_ratio[220,0]
zsteps = 100

# 5 and 80 are arbitrary numbers, I'm trying to somewhat preserve the z grid spacing
z_left = z = np.linspace(0.001, 0.009, 5)
z_right = z = np.linspace(2.20, 4, 80)


z_center = np.linspace(0.01, 2.20, zsteps)

L_ratio_array = np.asarray([L_ratio(zi) for zi in z_center])
L_ratio_left_array  = np.asarray([extrapolate(lumin_ratio, zi, selector="left") for zi in z_left])
L_ratio_right_array = np.asarray([extrapolate(lumin_ratio, zi, selector="right") for zi in z_right])

# plt.plot(z_center, L_ratio_array, ".-")
# plt.plot(z_left,  L_ratio_left_array, ".-")
# plt.plot(z_right, L_ratio_right_array, ".-")


lumin_ratio_2 = np.zeros((221+5+80,2))
lumin_ratio_2[:5,0] = z_left
lumin_ratio_2[5:221+5,0] = lumin_ratio[:,0]
lumin_ratio_2[221+5:,0] = z_right

lumin_ratio_2[:5,1] = L_ratio_left_array
lumin_ratio_2[5:221+5,1] = lumin_ratio[:,1]
lumin_ratio_2[221+5:,1] = L_ratio_right_array

plt.plot(lumin_ratio_2[:,0], lumin_ratio_2[:,1], ".-")

np.savetxt(r"%s\data\scaledmeanlum-E2Sa_EXTRAPOLATED.txt" %path, lumin_ratio_2)

