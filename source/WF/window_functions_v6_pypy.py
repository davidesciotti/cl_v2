import numpy as np
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp1d

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

def n_i(z,i):
    integrand   = lambda z_p, z : n(z) * pph(z,z_p)
    numerator   = quad(integrand, z_minus[i], z_plus[i], args = (z))
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])
#    denominator = nquad(integrand, [[z_minus[i], z_plus[i]],[z_min, z_max]] )
    return numerator[0]/denominator[0]

############ section to delete: checking if ni(z) is the issue
#niz_import = np.genfromtxt("%s/data/dati_vincenzo/niNorm.dat" %path)
#def n_i(z,i):
#    # it's i+1: first column is for the redshift array
#    niz_interp = interp1d(niz_import[:,0], niz_import[:,i+1], kind = "linear")
#    result_array = niz_interp(z) #z is considered as an array
#    result = result_array.item() #otherwise it would be a 0d array 
#    return result

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
    return b(i)*(n_i(z,i)/n_bar_i(i))*H0*E(z)/c
    # xxx I'm already dividing by c!

########################### modify WFs according to Overleaf draft
def wil_SSC(z,i):
    return wil_tot(z,i)/r(z)**2
    
def wig_SSC(z,i):
    return wig(z,i)/r(z)**2
###################################################








def compute():

    wil_tot_array = np.zeros((z.shape[0], zbins))
    wig_array     = np.zeros((z.shape[0], zbins))
    wil_SSC_array = np.zeros((z.shape[0], zbins))
    wig_SSC_array = np.zeros((z.shape[0], zbins))
    niz_array     = np.zeros((z.shape[0], zbins))
    # calling the functions and saving the results:
    D_array   = np.asarray([D(zi) for zi in z])
    
    for i in range(zbins):
        wil_tot_array[:,i] = np.asarray([wil_tot(zi, i) for zi in z])  
        wig_array[:,i]     = np.asarray([wig(zi, i) for zi in z])
        
        wil_SSC_array[:,i] = np.asarray([wil_SSC(zi,i) for zi in z])
        wig_SSC_array[:,i] = np.asarray([wig_SSC(zi,i) for zi in z])
            
        niz_array[:,i]     = np.asarray([n_i(zi, i) for zi in z])
        
        print("debug: I'm working the column number %i" %i)
        end = time.time()
        print("the program took %i seconds to run" %(end - start))
    
    return(wil_tot_array, wig_array, wil_SSC_array, wig_SSC_array, niz_array, D_array) 

def insert_zColumn(wil_tot_array, wig_array, wil_SSC_array, wig_SSC_array, niz_array, D_array):
    D_array = np.reshape(D_array, (z.shape[0], 1))
    
    wil_tot_array = np.insert(wil_tot_array, 0, z, axis=1)
    wig_array     = np.insert(wig_array, 0, z, axis=1)
    wil_SSC_array = np.insert(wil_SSC_array, 0, z, axis=1)
    wig_SSC_array = np.insert(wig_SSC_array, 0, z, axis=1)
    niz_array     = np.insert(niz_array, 0, z, axis=1)
    D_array       = np.insert(D_array, 0, z, axis=1)

def save():
    np.save("%s/output/wil_tot.npy" %path, wil_tot_array)
    np.save("%s/output/wig.npy"     %path, wig_array)
    np.save("%s/output/wil_SSC.npy" %path, wil_SSC_array)
    np.save("%s/output/wig_SSC.npy" %path, wig_SSC_array)
    np.save("%s/output/n_i(z).npy"  %path, niz_array)
    np.save("%s/output/D(z).npy"    %path, D_array)
    np.save("%s/output/z.npy"       %path, z)
    
    
########################### computing and saving the functions


############ sylvain
#niz_sylvain  = np.genfromtxt("%s/data/windows_sylvain/nz_source/bin_%i.txt" %(path, i+1))
#z_sylvain    = np.genfromtxt("%s/data/windows_sylvain/nz_source/z.txt" %(path))

######### vincenzo
niz_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/niNorm.dat" %path)
wil_SSC_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiL.dat" %path)
wig_SSC_vincenzo = np.genfromtxt("%s/data/dati_vincenzo/WiG.dat" %path)
rz_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/ComDist-LCDM-NonLin-eNLA.dat" %path)
L_ratio_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/scaledmeanlum-E2Sa.dat" %path)
wil_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/WiGamma-LCDM-NonLin-eNLA.dat" %path)
wig_vincenzo_2 = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/WiG-LCDM-NonLin-eNLA.dat" %path)

######## davide
zsteps = 20
z = np.linspace(z_min, z_max, zsteps)
i=0
#niz_davide = np.asarray([n_i(zi, i) for zi in z])
#pph_array  = np.asarray([pph(zi, zp) for zi in z])
#nz_array   = np.asarray([n(zi) for zi in z])
#wil_SSC_davide = np.asarray([wil_SSC(zi, 0) for zi in z])
#rz_davide = np.asarray([r(zi) for zi in z])
#wig_SSC_davide = np.asarray([wig_SSC(zi, i) for zi in z])
#L_ratio_davide = np.asarray([L_ratio(zi) for zi in z])
#wil_davide = np.asarray([wil(zi, i) for zi in z])
wig_davide = np.asarray([wig(zi, i) for zi in z])


####### saving txts
#np.savetxt("%s/output/prove_varie/wil_SSC_davide_%i.txt" %(path,i), wil_SSC_davide)
#np.savetxt("%s/output/prove_varie/r(z).txt" %(path), rz_davide)
#np.savetxt("%s/output/prove_varie/L_ratio.txt" %(path), L_ratio_davide)




#wil_tot_array, wig_array, wil_SSC_array, wig_SSC_array, niz_array, D_array = compute()
#insert_zColumn(wil_tot_array, wig_array, wil_SSC_array, wig_SSC_array, niz_array, D_array)
#save()



end = time.time()
print("the program took %i seconds to run" %(end - start))


