import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
#from functools import partial
#from astropy.cosmology import WMAP9 as cosmo
#from astropy import constants as const
#from astropy import units as u

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"
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
    
def pph(z, z_p):    
    first_addendum  = (1-f_out)/(np.sqrt(2*np.pi)*sigma_b*(1+z)) * \
                      np.exp(-0.5* ( (z-c_b*z_p-z_b)/(sigma_b*(1+z)) )**2)
    second_addendum = (f_out)/(np.sqrt(2*np.pi)*sigma_o*(1+z)) * \
                      np.exp(-0.5* ( (z-c_o*z_p-z_o)/(sigma_o*(1+z)) )**2)      
    return first_addendum + second_addendum

def n(z):
    result = (z/z_0)**2 * np.exp(-(z/z_0)**(3/2))
#    result = result/0.424264 # normalising the distribution ? xxx
    return result

def n_i_numerator(z, i):
    integrand = lambda z_p, z : n(z) * pph(z,z_p)
    numerator = quad(integrand, z_minus[i], z_plus[i], args = (z))
    return numerator

def n_i_denominator(i):
    integrand = lambda z_p, z : n(z) * pph(z,z_p)
    # xxx the order is important: first the x bounds, then the y bounds
    # with f(y,x) being the integrand
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])
    return denominator

def n_i(z,i):
    num = n_i_numerator(z,i)
    den = n_i_denominator(i)
    return num[0]/den[0] # attention! quad returns a tuple

def wil_tilde(z, i): #xxx attention, check carefully
    integrand = lambda z_prime, z, i: n_i(z_prime, i)*(1 - r_tilde(z)/r_tilde(z_prime))
    # integrate in z_prime, it must be the first argument
    result = quad(integrand, z, z_max, args = (z, i))
    return result[0]
    
def wil(z,i):
    return (3/2)*(H0/c)*Om0*(1+z)*r_tilde(z)*wil_tilde(z,i)

########################################################### IA

def W_IA(z,i):
    result = (H0/c)*n_i(z,i)*E(z)
    return result

lumin_ratio = np.genfromtxt("%s/data/scaledmeanlum-E2Sa.dat" %path)
def L_ratio(z):
    z=1
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

def n_bar_i(i):
    result = quad(n_i, z_min, z_max, args = i) 
    return result[0]
    
def wig(z,i):
    return b(i)*(n_i(z,i)/n_bar_i(i))*H0*E(z)

########################### end of function declaration

######## calling the functions and saving the results:
z = np.linspace(0.001, 3, 100)

# arrays in which to save the WFs
wil_toSave = np.zeros((z.shape[0], 11))
wig_toSave = np.zeros((z.shape[0], 11))

# setting the first column
wil_toSave[:,0] = wig_toSave[:,0] = z


column = 0
# setting the remaining columns
#for column in range(1,11):
    # wil_tot (with IA)
    # calling the functions for every zi in z
WF_lens = np.asarray([wil_tot(zi, column) for zi in z])     
#ni_z_array = np.asarray([n_i(zi, column) for zi in z])   
#np.savetxt("%s/output/n_%i(z).txt" %(path, column), ni_z_array)
wil_toSave[:, column] = WF_lens 
#    print("wil done")
    # wig
#    WF_gal = np.asarray([wig(zi, column) for zi in z])
#    wig_toSave[:, column] = WF_gal
#    print("wil done")
#    print("debug: I'm working the column number %i" %column)
    
end = time.time()
print("the program took %i seconds to run" %(end - start))
    
    
np.save("%s/output/wil_tot.npy" %path, wil_toSave)
np.save("%s/output/wig.npy" %path, wig_toSave)




# alternative way
#z = np.linspace(1e-3, 5, 300)
#vec_int = np.vectorize(n_i())
#plt.plot(z, vec_int(z))

# yet another way
#def r_tilde(z):
#    result = np.array(
#        list(map(partial(quad, E, 0), z))
#    )[:, 0]  # integrate E(z) from 0 to z
##    result = result*c/H0 # delete if you want r_tilde, see formula (104)
#    return result


#cosmo.comoving_distance(z)  
#fig, ax = plt.subplots()
#ax.plot(z, r(z), "r--")
#ax.plot(z, cosmo.comoving_distance(z)  )
#ax.plot(z_p, pph(z_p) )
#ax.plot(z, n(z) )
#fig.show()

#residuals = (r(z)*u.dimensionless_unscaled - cosmo.comoving_distance(z)*u.dimensionless_unscaled) / r(z) * 100
#plt.plot(z, residuals, "g")