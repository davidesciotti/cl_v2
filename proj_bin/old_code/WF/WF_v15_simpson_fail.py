import numpy as np
from scipy.integrate import quad, dblquad, nquad, simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from numba import jit, njit
import sys
from pathlib import Path
import matplotlib

matplotlib.use('Qt5Agg')

# get project directory
path = Path.cwd().parent.parent
# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config

sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

script_name = sys.argv[0]
params = {'lines.linewidth': 2.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral'
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################


# from functools import partial
# from astropy.cosmology import WMAP9 as cosmo
# from astropy import constants as const
# from astropy import units as u
start = time.perf_counter()

WFs_input_folder = "WFs_v7_zcut_noNormalization"
WFs_output_folder = "WFs_v15_zNLA_gen22"

# saving the options (ooo) in a text file:
with open("%s/output/WF/%s/options.txt" % (path, WFs_output_folder), "w") as text_file:
    print("zcut: yes \nnbar normalization: yes \nn(z) normalization: no \nbias: multi-bin \nniz: davide",
          file=text_file)

# interpolating to speed up
# with z cut following Vincenzo's niz
# with n_bar normalisation
# with "multi-bin" bias
# with niz from Vincenzo

# FIXME: computing the PySSC WFs in this way (ie recomputing them and not just
# fividing the ones you have by r^2(z)) is super inefficient! fix this, very easyà


# define the name of the directory to be created
# new_folder = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/base_functions_v5"
# new_folder = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v5"
# try:
#     os.mkdir(new_folder)
# except OSError:
#     print ("Creation of the directory %s failed" % new_folder)
# else:
#     print ("Successfully created the directory %s " % new_folder)


c = 299792.458  # km/s
H0 = 67  # km/(s*Mpc)

Om0 = 0.32
Ode0 = 0.68
Ox0 = 0
gamma = 0.55

z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))

z_mean = (z_plus + z_minus) / 2
z_min = z_minus[0]
z_max = z_plus[9]
# xxx is z_max = 4 to be used everywhere?
z_max = 4

f_out = 0.1
sigma_b = 0.05
sigma_o = 0.05
c_b = 1.0
c_o = 1.0
z_b = 0
z_o = 0.1

z_m = 0.9
z_0 = z_m / np.sqrt(2)

A_IA = 1.72
C_IA = 0.0134
eta_IA = -0.41
beta_IA = 2.17
# beta_IA = 0.0

zbins = 10

simps_z_step_size = 1e-4

n_bar = np.genfromtxt("%s/output/n_bar.txt" % path)


####################################### function definition
# @njit
def E(z):
    result = np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
    return result


def inv_E(z):
    result = 1 / np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
    return result


def r_tilde_original(z):
    # r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    result = quad(inv_E, 0, z)  # integrate 1/E(z) from 0 to z
    return result[0]


def r_original(z):
    result = c / H0 * quad(inv_E, 0, z)[0]
    return result


z_simps = np.linspace(z_min, z_max, 10_000)
integrand = 1 / E(z_simps)


def r_tilde(z, integrand=integrand):

    # r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    # populate the array with the values of r_tilde(z)

    # populate integrand - no need for a for loop!!

    # integrand = 1 / E(z)

    # find index closest to z
    z_index = np.argmin(np.abs(z_simps - z))
    # print(z, z_simpson[z_index+1])

    integration_domain = z_simps[:z_index + 1]  # XXX i'm here

    integrand = integrand[:z_index + 1]

    # print(integrand.shape, integration_domain.shape)

    result = simps(integrand, integration_domain)

    # result = quad(inv_E, 0, z)  # integrate 1/E(z) from 0 to z
    return result

WF_samples = 5_000
simps_samples = 10_000
z_matrix = np.zeros((WF_samples, simps_samples))
z_WF = np.linspace(z_min, z_max, WF_samples)
z_simps = np.linspace(z_min, z_max, simps_samples)
zz_WF, zz_simps = np.meshgrid(z_WF, z_simps)
integrand = 1 / E(zz_WF)

for zi in range(WF_samples):
    z_matrix[zi, :] = np.linspace(z_min, z_WF[zi], simps_samples)


# test
start = time.perf_counter()
old = np.asarray([r_tilde_original(z) for z in z_simps])
print("original: %.2f" % (time.perf_counter() - start))

start = time.perf_counter()
integrand = 1 / E(z_matrix)
new = simps(integrand, z_matrix, axis=0)
# new = np.asarray([r_tilde(z) for z in z_simps])
print("new: %.2f" % (time.perf_counter() - start))

diff = (old / new - 1) * 100
plt.plot(z_simps, diff)
plt.yscale('log')

@njit
def generate_grid(z_arr_WF, trapz_samples):
    z_trapz_mesh = np.zeros((z_arr_WF.shape[0], trapz_samples))
    for zi, zval in enumerate(z_arr_WF):
        z_trapz_mesh[zi, :] = np.linspace(0, zval, trapz_samples)
    return z_trapz_mesh

z_arr_WF = np.linspace(z_min, z_max, 50_000)
trapz_samples = 500

# time this
start = time.perf_counter()
old = np.asarray([r_tilde(z) for z in z_arr_WF])
print("old: ", time.perf_counter() - start)

start = time.perf_counter()
z_trapz_mesh = generate_grid(z_arr_WF, trapz_samples)
integrand = 1 / E(z_trapz_mesh)
integral = simps(integrand, z_trapz_mesh, axis=1)
print("new: ", time.perf_counter() - start)

plt.plot(z_arr_WF, old, label="old")
plt.plot(z_arr_WF, new, label="new")


assert 1 > 2


def r(z):
    result = c / H0 * quad(inv_E, 0, z)[0]
    return result


def pph(z, z_p):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_b * (1 + z)) * \
                     np.exp(-0.5 * ((z - c_b * z_p - z_b) / (sigma_b * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_o * (1 + z)) * \
                      np.exp(-0.5 * ((z - c_o * z_p - z_o) / (sigma_o * (1 + z))) ** 2)
    return first_addendum + second_addendum


def n(z):  # note: if you import n_i(z) this function doesn't get called!
    result = (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))
    # TODO normalize the distribution or not?
    # result = result*(30/0.4242640687118783) # normalising the distribution?
    return result


################################## niz ##############################################

# choose the cut XXX
# n_i_import = np.genfromtxt("%s/data/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path) # vincenzo (= davide standard, pare)
# n_i_import = np.genfromtxt(path.parent / "common_data/vincenzo/14may/InputNz/niTab-EP10-RB.dat") # vincenzo, more recent (= davide standard, anzi no!!!!)
# n_i_import = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v3_cut/niz_e-19cut.txt") # davide e-20cut
n_i_import = np.load("%s/output/WF/WFs_v2/niz.npy" % path)  # davide standard


# n_i_import_2 = np.genfromtxt("%s/output/WF/%s/niz.txt" %(path, WFs_input_folder)) # davide standard with zcutVincenzo


def n_i(z, i):
    n_i_interp = interp1d(n_i_import[:, 0], n_i_import[:, i + 1], kind="linear")
    result_array = n_i_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


# note: the normalization of n(z) should be unimportant, here I compute a ratio
# where n(z) is present both at the numerator and denominator!
# as a function, including (of not) the ie-20 cut
# def n_i(z,i):
#     integrand   = lambda z_p, z : n(z) * pph(z,z_p)
#     numerator   = quad(integrand, z_minus[i], z_plus[i], args = (z))
#     denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])
# #    denominator = nquad(integrand, [[z_minus[i], z_plus[i]],[z_min, z_max]] )
# #    return numerator[0]/denominator[0]*3 to have quad(n_i, 0, np.inf = nbar_b/20 = 3)
#     result = numerator[0]/denominator[0]
#     # if result < 6e-19: # remove these 2 lines if you don't want the cut
#     #     result = 0
#     return result


################################## end niz ##############################################


def wil_tilde_old(z, i):  # xxx attention, check carefully
    integrand = lambda z_prime, z, i: n_i(z_prime, i) * (1 - r_tilde(z) / r_tilde(z_prime))
    # integrate in z_prime, it must be the first argument
    result = quad(integrand, z, z_max, args=(z, i), limit=150)
    return result[0]


def wil_tilde(z, i):  # xxx attention, check carefully
    # integrand = lambda z_prime, z, i: n_i(z_prime, i) * (1 - r_tilde(z) / r_tilde(z_prime))

    # populate array with the integrand
    integrand_array = np.zeros(z_array.size)
    for z_prime_idx, z_prime in enumerate(z_array):
        integrand_array[z_prime_idx] = n_i(z_prime, i) * (1 - r_tilde(z) / r_tilde(z_prime))

    # integrate with simpson's rule
    result = simps(integrand_array, z_array)

    return result


def wil_noIA_IST(z, i):
    return (3 / 2) * (H0 / c) * Om0 * (1 + z) * r_tilde(z) * wil_tilde(z, i)


########################################################### IA

def W_IA(z, i):
    result = (H0 / c) * n_i(z, i) * E(z)
    return result


lumin_ratio = np.genfromtxt("%s/data/scaledmeanlum-E2Sa_EXTRAPOLATED.txt" % path)


def L_ratio(z):
    lumin_ratio_interp1d = interp1d(lumin_ratio[:, 0], lumin_ratio[:, 1], kind='linear')
    result_array = lumin_ratio_interp1d(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array OK!
    return result


def F_IA(z):
    result = (1 + z) ** eta_IA * (L_ratio(z)) ** beta_IA
    return result


# use formula 23 for Om(z)
def Om(z):
    return Om0 * (1 + z) ** 3 / E(z) ** 2


def D(z):
    integrand = lambda x: Om(x) ** gamma / (1 + x)
    integral = quad(integrand, 0, z)[0]
    result = np.exp(-integral)
    return result


def IA_term(z, i):
    return (A_IA * C_IA * Om0 * F_IA(z)) / D(z) * W_IA(z, i)


def wil_IA_IST(z, i):
    return wil_noIA_IST(z, i) - IA_term(z, i)


###################### wig ###########################
def b(i):
    return np.sqrt(1 + z_mean[i])


def b_new(z):
    for i in range(zbins):
        if z_minus[i] <= z and z < z_plus[i]:
            return b(i)
        if z > z_plus[-1]:  # max redshift bin
            return b(9)


# debug
# plt.plot(z_mean, b(range(10)), "o-", label = "b_old" )
# z = np.linspace(0, 4, 300)
# array = np.asarray([b_new(zi) for zi in z])
# plt.plot(z, array, ".-", label = "b_new" )
# print(array)

# I have saved the results of this function in the array n_bar[i].
# no need to call it again. ooo
def n_bar_i(i):
    result = quad(n_i, z_min, z_max, args=i, limit=100)
    return result[0]


def wig_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    return b(i) * (n_i(z, i) / n_bar[i]) * H0 * E(z) / c


def wig_multiBinBias_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    # print(b_new(z), z) # debug
    return b_new(z) * (n_i(z, i) / n_bar[i]) * H0 * E(z) / c


def wig_noBias_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less) ooo
    return (n_i(z, i) / n_bar[i]) * H0 * E(z) / c


# def wig_IST(z,i): # without n_bar normalisation
#     return b(i) * n_i(z,i) *H0*E(z)/c
# xxx I'm already dividing by c!

########################### modify WFs according to Overleaf draft to prepare them for PySSC
def wil_noIA_PySSC(z, i):
    return wil_noIA_IST(z, i) / r(z) ** 2


def wil_IA_PySSC(z, i):
    return wil_IA_IST(z, i) / r(z) ** 2


def wig_PySSC(z, i):
    return wig_IST(z, i) / r(z) ** 2


def wig_noBias_PySSC(z, i):
    return wig_noBias_IST(z, i) / r(z) ** 2


def wig_multiBinBias_PySSC(z, i):
    return wig_multiBinBias_IST(z, i) / r(z) ** 2


###################################################


def compute_1D(function, z_array_toInsert):
    array = np.zeros((z.shape[0], 2))  # it has only one column apart from z
    array[:, 0] = z_array_toInsert
    array[:, 1] = np.asarray([function(zi) for zi in z])
    return array


# @jit(nopython = True)
def compute_2D(function, z_array_toInsert):
    array = np.zeros((z.shape[0], zbins + 1))
    array[:, 0] = z_array_toInsert

    for i in range(zbins):
        array[:, i + 1] = np.asarray([function(zi, i) for zi in z])
        # alternative
        # array[:,i+1] = np.vectorize(function)(z,i)

        print("debug: I'm working on column number %i" % i)
    end = time.perf_counter()
    print("the function took %i seconds to run" % (end - start))
    return array


def save(array, name):
    np.savetxt(f"{path}/output/WF/{WFs_output_folder}/{name}.txt", array)


###############################################################################
###################### END OF FUNCTION DEFINITION #############################
###############################################################################

########################### computing and saving the functions
zpoints = 1_000

z = np.linspace(z_min, 4, zpoints)

z_array = np.linspace(z_min, z_max, zpoints)
# using Sylvain's z
# z = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/windows_sylvain/nz_source/z.txt")


############ WiG

# array_appoggio = compute_2D(wig_IST, z)
# name = f"wig_davide_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_IST DONE")

# array_appoggio = compute_2D(wig_PySSC, z)
# name = f"wig_davide_PySSC_nz{zsteps}"
# save(array_appoggio, name)
# print("wig_PySSC DONE")

# array_appoggio = compute_2D(wig_noBias_IST, z)
# name = f"wig_davide_noBias_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_noBias_IST DONE")

# array_appoggio = compute_2D(wig_noBias_PySSC, z)
# name = f"wig_davide_noBias_PySSC_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_davide_noBias_PySSC DONE")

"""
array_appoggio = compute_2D(wig_multiBinBias_IST, z)
name = f"wig_davide_multiBinBias_IST_nz{zsteps}" 
save(array_appoggio, name)
print("wig_multiBinBias_IST DONE")
"""

# array_appoggio = compute_2D(wig_multiBinBias_PySSC, z)
# name = f"wig_davide_multiBinBias_PySSC_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_multiBinBias_PySSC DONE")

############ WiL   

# array_appoggio = compute_2D(wil_noIA_IST, z)
# name = f"wil_davide_noIA_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wil_noIA_IST DONE")

print('working')

start = time.perf_counter()
array_appoggio = compute_2D(wil_IA_IST, z)
name = f"wil_davide_IA_IST_nz{zpoints}_bia{beta_IA:.2f}"
# save(array_appoggio, name)
print("wil_IA_IST DONE")
print("the function took %.2f seconds to run" % (time.perf_counter() - start))

# array_appoggio = compute_2D(wil_noIA_PySSC, z)
# name = f"wil_davide_noIA_PySSC_nz{zsteps}" 
# save(array_appoggio, name)
# print("wil_noIA_PySSC DONE")

# array_appoggio = compute_2D(wil_IA_PySSC, z)
# name = f"wil_davide_IA_PySSC_nz{zsteps}" 
# save(array_appoggio, name)
# print("wil_IA_PySSC DONE")

########## others
# array_appoggio = compute_1D(L_ratio, z)
# np.savetxt("%s/output\%s/L_ratio.txt" %(path, WFs_output_folder), array_appoggio)
# print("L_ratio DONE")
# array_appoggio = compute_2D(n_i, z)
# np.savetxt("%s/output\%s/niz_e-19cut.txt" %(path, WFs_output_folder), array_appoggio)
# print("niz DONE")
# the ones I need to send to Sylvain:
# array_appoggio = compute_2D(W_IA, z)
# save(array_appoggio, "W_IA")
# print("W_IA DONE")
# array_appoggio = compute_2D(IA_term, z)
# save(array_appoggio, "IA_term")
# print("IA_term DONE")    
# array_appoggio = compute_2D(wil_tilde, z)
# save(array_appoggio, "wil_tilde")
# print("wil_tilde DONE")
# compute and save niz, discard 
# array_appoggio = compute_2D(n_i, z)
# save(array_appoggio, "niz")
# print("niz DONE")

print("the script took %i seconds to run" % (time.perf_counter() - start))

# TODO IA DEBUGGING
# wil_davide = np.genfromtxt("%s\output\%s\wil.txt" %(path, WFs_output_folder))
# wil_tot_davide = np.genfromtxt("%s\output\%s\wil_tot.txt" %(path, WFs_output_folder))
# IA_term_array = compute_2D(IA_term, z)
# W_IA_array = compute_2D(W_IA, z)
# diff = np.abs(wil_davide - wil_tot_davide) / wil_davide * 100

# column = 5
# plt.plot(wil_davide[:,0], wil_davide[:,column + 1], label="wil")
# plt.plot(wil_tot_davide[:,0], wil_tot_davide[:,column + 1], label="wil_tot = wil - IA_term")
# plt.plot(IA_term_array[:,0], IA_term_array[:,column + 1], label="IA_term")
# # plt.plot(z, diff[:,column + 1], label="diff")
# # plt.hlines(5, 0, 4, "red", label = "5 %%")
# # plt.plot(W_IA_array[:,0], W_IA_array[:,column + 1], label="W_IA_array")
# plt.legend(prop={'size': 14})
# plt.title("$W_i^L(z)$, i = %i, IA vs non-IA" % column)
# plt.ylabel("$W_i^L(z)$")
# plt.xlabel("$z$")
# plt.grid()
# plt.tight_layout()
# plt.yscale("log")

# fig, axes = plt.subplots(2, 1, figsize=(10,4))
# axes[0].plot(wil_davide[:,0], wil_davide[:,column + 1], label="wil")
# axes[0].plot(wil_tot_davide[:,0], wil_tot_davide[:,column + 1], label="wil_tot")
# axes[0].plot(W_IA_array[:,0], W_IA_array[:,column + 1], label="W_IA_array")
# axes[0].set_title("Normal scale")
# axes[0].legend()
# plt.xlabel("$z$")
# plt.ylabel("WiL")

# axes[1].plot(wil_davide[:,0], wil_davide[:,column + 1], label="wil")
# axes[1].plot(wil_tot_davide[:,0], wil_tot_davide[:,column + 1], label="wil_tot")
# axes[1].plot(W_IA_array[:,0], W_IA_array[:,column + 1], label="W_IA_array")
# axes[1].set_yscale("log") 
# axes[1].set_title("Logarithmic scale (y)")
# axes[1].legend()
# plt.xlabel("$z$")
# plt.ylabel("WiL")
# plt.tight_layout()
