import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpire import WorkerPool
from numba import njit
from scipy.integrate import quad, quad_vec, simpson
from scipy.interpolate import interp1d, interp2d

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent
sys.path.append(str(project_path))

import lib.cosmo_lib as csmlb
import config.config as cfg

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
####################### ########################################################


start = time.perf_counter()

WFs_input_folder = "WFs_v7_zcut_noNormalization"
WFs_output_folder = "WFs_v16_eNLA_may22"

# saving the options (ooo) in a text file:
with open("%s/output/WF/%s/options.txt" % (project_path, WFs_output_folder), "w") as text_file:
    print("zcut: yes \nnbar normalization: yes \nn(z) normalization: no \nbias: multi-bin \nniz: davide",
          file=text_file)

# interpolating to speed up
# with z cut following Vincenzo's niz
# with n_bar normalisation
# with "multi-bin" bias
# with niz from Vincenzo


# define the name of the directory to be created
# new_folder = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/base_functions_v5"
# new_folder = "C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v5"
# try:
#     os.mkdir(new_folder)
# except OSError:
#     print ("Creation of the directory %s failed" % new_folder)
# else:
#     print ("Successfully created the directory %s " % new_folder)


c = cfg.c
H0 = cfg.H0
Om0 = cfg.Om0
Ode0 = cfg.Ode0
Ob0 = cfg.Ob0
Ox0 = cfg.Ox0
gamma = cfg.gamma

z_minus = cfg.z_minus
z_plus = cfg.z_plus
z_mean = cfg.z_mean
z_min = cfg.z_min
z_max = cfg.z_max
z_m = cfg.z_m
z_0 = cfg.z_0
zbins = cfg.zbins

f_out = cfg.f_out
sigma_b = cfg.sigma_b
sigma_o = cfg.sigma_o
c_b = cfg.c_b
c_o = cfg.c_o
z_b = cfg.z_b
z_o = cfg.z_o

A_IA = cfg.A_IA
C_IA = cfg.C_IA
eta_IA = cfg.eta_IA
beta_IA = cfg.beta_IA

simps_z_step_size = 1e-4

n_bar = np.genfromtxt("%s/output/n_bar.txt" % project_path)
lumin_ratio = np.genfromtxt("%s/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt" % project_path)


####################################### function definition


@njit
def pph(z, z_p):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_b * (1 + z)) * \
                     np.exp(-0.5 * ((z - c_b * z_p - z_b) / (sigma_b * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_o * (1 + z)) * \
                      np.exp(-0.5 * ((z - c_o * z_p - z_o) / (sigma_o * (1 + z))) ** 2)
    return first_addendum + second_addendum


@njit
def n(z):  # note: if you import n_i(z) this function doesn't get called!
    result = (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))
    # TODO normalize the distribution or not?
    # result = result*(30/0.4242640687118783) # normalising the distribution?
    return result


################################## niz ##############################################

# choose the cut XXX
# TODO re-compute and check this!
# n_i_import = np.genfromtxt("%s/input/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path) # vincenzo (= davide standard, pare)
# n_i_import = np.genfromtxt(path.parent / "common_data/vincenzo/14may/InputNz/niTab-EP10-RB.dat") # vincenzo, more recent (= davide standard, anzi no!!!!)
# n_i_import = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v3_cut/niz_e-19cut.txt") # davide e-20cut
n_i_import = np.load("%s/output/WF/WFs_v2/niz.npy" % project_path)  # davide standard


# n_i_import_2 = np.genfromtxt("%s/output/WF/%s/niz.txt" %(path, WFs_input_folder)) # davide standard with zcutVincenzo


def n_i_old(z, i):
    n_i_interp = interp1d(n_i_import[:, 0], n_i_import[:, i + 1], kind="linear")
    result_array = n_i_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


z_values_from_nz = n_i_import[:, 0]
i_array = np.asarray(range(zbins))
n_i_import_cpy = n_i_import.copy()[:, 1:]  # remove redshift column

# note: this is NOT an interpolation in i, which are just the bin indices and will NOT differ from the values 0, 1, 2
# ... 9. The purpose is just to have a 2D vectorized callable.
n_i_new = interp2d(i_array, z_values_from_nz, n_i_import_cpy, kind="linear")


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


# @njit
def wil_tilde_integrand_old(z_prime, z, i):
    return n_i_old(z_prime, i) * (1 - csmlb.r_tilde(z) / csmlb.r_tilde(z_prime))


def wil_tilde_old(z, i):
    # integrate in z_prime, it must be the first argument
    result = quad(wil_tilde_integrand_old, z, z_max, args=(z, i))
    return result[0]


# version with quad vec, very slow, I don't know why. It is the i_array that is vectorized, because z_prime is integrated over
def wil_tilde_integrand_new(z_prime, z, i_array):
    return n_i_new(i_array, z_prime).T * (1 - csmlb.r_tilde(z) / csmlb.r_tilde(z_prime))


def wil_tilde_new(z, i_array):
    # integrate in z_prime, it must be the first argument
    return quad_vec(wil_tilde_integrand_new, z, z_max, args=(z, i_array))[0]

# z = 2.00000000e-03
# start = time.perf_counter()
# old = [wil_tilde_old(z, i) for i in range(zbins)]
# print("old:", time.perf_counter() - start)
#
# start = time.perf_counter()
# new = wil_tilde_new(z, i_array)
# print("new:", time.perf_counter() - start)
#
# start = time.perf_counter()
# simps = wil_tilde_new(z, i_array)
# print("simps:", time.perf_counter() - start)
#
# assert 1 > 2

# TEST
z_array_quad = np.linspace(0.002, 3.9, 50)
z_array = np.linspace(0.002, 3.9, 500)
# z_array = np.logspace(np.log10(0.002), np.log10(3.9), 10)
z_prime_array = z_array.copy()

# time it
start = time.perf_counter()
integrand = np.zeros((z_prime_array.size, z_array.size, zbins))
for z_idx, z_val in enumerate(z_array):
    # output order of wil_tilde_integrand_new is: z_prime, i
    integrand[:, z_idx, :] = wil_tilde_integrand_new(z_prime_array, z_val, i_array).T
print('integrand with for loop filled in: ', time.perf_counter() - start)

# for z_idx in range(z_array.size)[::5]:
#     z_prime_val = z_prime_array[z_idx]
#     plt.plot(z_array, wil_tilde_integrand_old(z_prime_val, z_array, i=i))
#     plt.plot(z_array, integrand[z_idx, :, i], '--')


# ! compare integrands
# integrand_old = np.zeros(integrand.shape)
# for z_prime_idx, z_prime in enumerate(z_prime_array):
#     for z_idx, z in enumerate(z_array):
#         for i in range(zbins):
#             integrand_old[z_prime_idx, z_idx, i] = wil_tilde_integrand_old(z_prime, z, i)
# print('done')

# np.allclose(integrand, integrand_old, rtol=1e-05)


start = time.perf_counter()
wil_tilde_simps = np.asarray([simpson(integrand[z_idx:, z_idx, :], z_array[z_idx:], axis=0) for z_idx, _ in enumerate(z_array)])
# wil_tilde_simps = np.asarray([np.trapz(integrand[z_idx:, z_idx, :], axis=0) for z_idx, _ in enumerate(z_array)])
print('simpson integral done in: ', time.perf_counter() - start)

# with parallel:
# results_array = np.zeros((z_array.size, zbins))
# start = time.perf_counter()
# for z_idx, z in enumerate(z_array):
#     data = [(z, i) for i in range(zbins)]
#     with WorkerPool() as pool:
#         results = pool.map(wil_tilde_old, data, progress_bar=True)
#     results_array[z_idx, :] = np.asarray(results)
# print('with parallel computing: ', time.perf_counter() - start)


# TODO add check, i in niz must be an int, otherwise the function gets interpolated!!

# start = time.perf_counter()
# wil_tilde_old_arr = np.asarray([wil_tilde_old(z, i) for z in z_array_old for i in range(zbins)]).reshape(z_array_old.size, zbins)
# print('old done in', time.perf_counter() - start, 'seconds')

i = 0

start = time.perf_counter()
wil_tilde_old_arr = np.asarray([wil_tilde_old(z, i) for z in z_array_quad])
print('quad for only one bin done in ', time.perf_counter() - start, 'seconds')

# diff = (wil_tilde_simps[:, i]/wil_tilde_old_arr - 1) * 100

# start = time.perf_counter()
# assert i_array.dtype == np.dtype('int64')  # otherwise it interpolates!
# wil_tilde_new_result = np.asarray([wil_tilde_new(z, i_array) for z in z_array_quad])
# print('quad interm, for all bins done in ', time.perf_counter() - start, 'seconds')

# assert i_array.dtype == np.dtype('int64')  # otherwise it interpolates!

plt.plot(z_array, wil_tilde_simps[:, i], label='simpson')
plt.plot(z_array_quad, wil_tilde_old_arr, '.-', label='old')
# plt.plot(z_array_quad, wil_tilde_new_arr[:, i, 0], '.-', label='interm')
# plt.plot(z_array, diff, '.-', label='perc diff')
plt.legend()
plt.grid()

assert 1 > 2



def wil_noIA_IST(z, i, wil_tilde_array):
    return (3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlb.r_tilde(z) * wil_tilde_array


########################################################### IA
# @njit
def W_IA(z, i):
    result = (H0 / c) * n_i(z, i) * csmlb.E(z)
    return result


# def L_ratio(z):
#     lumin_ratio_interp1d = interp1d(lumin_ratio[:, 0], lumin_ratio[:, 1], kind='linear')
#     result_array = lumin_ratio_interp1d(z)  # z is considered as an array
#     result = result_array.item()  # otherwise it would be a 0d array
#     return result

# test this
L_ratio = interp1d(lumin_ratio[:, 0], lumin_ratio[:, 1], kind='linear')


# @njit
def F_IA(z):
    result = (1 + z) ** eta_IA * (L_ratio(z)) ** beta_IA
    return result


# use formula 23 for Om(z)
# @njit
def Om(z):
    return Om0 * (1 + z) ** 3 / csmlb.E(z) ** 2


# @njit
def Dz_integrand(x):
    return Om(x) ** gamma / (1 + x)


def D(z):
    integral = quad(Dz_integrand, 0, z)[0]
    return np.exp(-integral)


# @njit
# def IA_term(z, i):
#     return (A_IA * C_IA * Om0 * F_IA(z)) / D(z) * W_IA(z, i)

# @njit
def IA_term(z, i, Dz_arr):
    return (A_IA * C_IA * Om0 * F_IA(z)) / Dz_arr * W_IA(z, i)


# @njit
def wil_IA_IST(z, i, wil_tilde_array, Dz_array):
    return wil_noIA_IST(z, i, wil_tilde_array[:, i]) - IA_term(z, i, Dz_array)


###################### wig ###########################
@njit
def b(i):
    return np.sqrt(1 + z_mean[i])


@njit
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
# def n_bar_i(i):
#     result = quad(n_i, z_min, z_max, args=i, limit=100)
#     return result[0]


@njit
def wig_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    return b(i) * (n_i(z, i) / n_bar[i]) * H0 * csmlb.E(z) / c


# @njit
def wig_multiBinBias_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    # print(b_new(z), z) # debug
    return b_new(z) * (n_i(z, i) / n_bar[i]) * H0 * csmlb.E(z) / c


@njit
def wig_noBias_IST(z, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less) ooo
    return (n_i(z, i) / n_bar[i]) * H0 * csmlb.E(z) / c


# def wig_IST(z,i): # without n_bar normalisation
#     return b(i) * n_i(z,i) *H0*csmlb.E(z)/c
# xxx I'm already dividing by c!

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
        start = time.perf_counter()
        array[:, i + 1] = np.asarray([function(zi, i) for zi in z])
        # alternative
        # array[:,i+1] = np.vectorize(function)(z,i)

        end = time.perf_counter()
        print(f"debug: I'm working on column number {i}, took {(end - start):.2f} seconds")

    return array


def save(array, name):
    np.savetxt(f"{project_path}/output/WF/{WFs_output_folder}/{name}.txt", array)


###############################################################################
###################### END OF FUNCTION DEFINITION #############################
###############################################################################

########################### computing and saving the functions
zpoints = 100

z = np.linspace(z_min, z_max, zpoints)

z_array = np.linspace(z_min, z_max, zpoints)
# using Sylvain's z
# z = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/input/windows_sylvain/nz_source/z.txt")


############ WiG

# array_appoggio = compute_2D(wig_IST, z)
# name = f"wig_davide_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_IST DONE")


# array_appoggio = compute_2D(wig_noBias_IST, z)
# name = f"wig_davide_noBias_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wig_noBias_IST DONE")


array_appoggio = compute_2D(wig_multiBinBias_IST, z)
name = f"wig_davide_multiBinBias_IST_nz{zpoints}"
save(array_appoggio, name)
print("wig_multiBinBias_IST DONE")

############ WiL

# array_appoggio = compute_2D(wil_noIA_IST, z)
# name = f"wil_davide_noIA_IST_nz{zsteps}" 
# save(array_appoggio, name)
# print("wil_noIA_IST DONE")

print('precomputing arrays un-vectorizable quantities (which use quad integration)')
start = time.perf_counter()
Dz_array = np.asarray([D(z) for z in z_array])
print('Dz_array done in ', time.perf_counter() - start, 'seconds')

i_array = np.asarray(range(zbins))
start = time.perf_counter()
wil_tilde_array = np.asarray([wil_tilde(z, i) for z in z_array for i in range(zbins)])
wil_tilde_array = np.asarray([wil_tilde(z, i_array) for z in z_array])
print('wil_tilde_array done in ', time.perf_counter() - start, 'seconds')

# TODO vectorize i...?
wil_IA_IST_arr = wil_IA_IST(z_array, 0, wil_tilde_array, Dz_array)

assert 1 > 2

start = time.perf_counter()
array_appoggio = compute_2D(wil_IA_IST, z)
name = f"wil_davide_IA_IST_nz{zpoints}_bia{beta_IA:.2f}"
# save(array_appoggio, name)
print("wil_IA_IST DONE in %.2f seconds" % (time.perf_counter() - start))

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
