import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from functools import partial
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as const
from astropy import units as u
import sympy

# obiettivo: formula 103 paper IST

c = 299792.458 # km/s
H0 = 67 #km/(s*Mpc)


def E(z):
    Om = 0.32
    Ode = 0.68
    Ox = 0
    result = 1/np.sqrt(Om*(1 + z)**3 + Ode + Ox*(1 + z)**2)
    return result

def r(z):
    result = np.array(
        list(map(partial(quad, E, 0), z))
    )[:, 0]  # integrate E(z) from 0 to z
    result = result*c/H0 #ok
    return result

def pph(z, z_p):
    f_out = 0.1
    sigma_b = 0.05
    sigma_o = 0.05
    c_b = 1.0
    c_o = 1.0
    z_b = 0
    z_o = 0.1
    
    first_addendum  = (1-f_out)/(np.sqrt(2*np.pi)*sigma_b*(1+z)) * \
                      np.exp(-0.5* ( (z-c_b*z_p-z_b)/(sigma_b*(1+z)) )**2)
    second_addendum = (f_out)/(np.sqrt(2*np.pi)*sigma_o*(1+z)) * \
                      np.exp(-0.5* ( (z-c_o*z_p-z_o)/(sigma_o*(1+z)) )**2)      
    return first_addendum + second_addendum

def n(z):
    z_m = 0.9
    z_0 = z_m/np.sqrt(2)
    result = (z/z_0)**2 * np.exp(-(z/z_0)**(3/2))
    return result

#def integrand(z_p, z):
#    return n(z) * pph(z, z_p)

def n_i(z, i):
    z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
    z_plus  = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))
    z_min = z_minus[0]
    z_max = z_plus[9]

    integrand = lambda z_p, z : n(z) * pph(z,z_p)

    numerator   =    quad(integrand, z_minus[i], z_plus[i], args = (z))
    denominator = dblquad(integrand, z_minus[i], z_plus[i], z_min, z_max)
    
    result = lambda z,i : numerator(z,i)/denominator(i)

    return result(z,i)

#
#z  = sympy.Symbol('z')
#z_p = sympy.Symbol('z_p')
#
#z_m = 0.9
#z_0 = z_m/np.sqrt(2)
#
#f_n   = (z/z_0)**2 * sympy.exp(-(z/z_0)**(3/2))
#f_out = 0.1
#sigma_b = 0.05
#sigma_o = 0.05
#c_b = 1.0
#c_o = 1.0
#z_b = 0
#z_o = 0.1
#
#first_addendum  = (1-f_out)/(sympy.sqrt(2*sympy.pi)*sigma_b*(1+z)) * \
#                  sympy.exp(-0.5* ( (z-c_b*z_p-z_b)/(sigma_b*(1+z)) )**2)
#second_addendum = (f_out)/(sympy.sqrt(2*sympy.pi)*sigma_o*(1+z)) * \
#                  sympy.exp(-0.5* ( (z-c_o*z_p-z_o)/(sigma_o*(1+z)) )**2)   
#g_pph = first_addendum + second_addendum
#
#sympy.integrate(f_n*g_pph, z_p)

z = np.linspace(1e-3, 5, 300)


vals = [n_i(zi,0) for zi in z]
vals = np.asarray(vals)
plt.plot(z, vals)

result = np.array(
    list(map(partial(quad, E, 0), z))
)[:, 0]  # integrate E(z) from 0 to z

#z = np.linspace(1e-3, 5, 300)
#vec_int = np.vectorize(n_i())
#plt.plot(z, vec_int(z))


cosmo.comoving_distance(z)  
fig, ax = plt.subplots()
ax.plot(z, r(z), "r--")
ax.plot(z, cosmo.comoving_distance(z)  )
ax.plot(z_p, pph(z_p) )
ax.plot(z, n(z) )
fig.show()

#residuals = (r(z)*u.dimensionless_unscaled - cosmo.comoving_distance(z)*u.dimensionless_unscaled) / r(z) * 100
#plt.plot(z, residuals, "g")