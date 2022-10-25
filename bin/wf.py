import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from camb.sources import GaussianSourceWindow
from numba import njit
from scipy.integrate import quad, quad_vec, simpson
from scipy.interpolate import interp1d, interp2d

project_path = Path.cwd().parent
home_path = Path.home()

# general libraries
sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTF
import mpl_cfg

# config files
sys.path.append(f'{project_path}/config')
import config.config as cfg

# project modules
# sys.path.append(f'{project_path}/bin')

# update plot pars
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')

###############################################################################
###############################################################################
###############################################################################


script_start = time.perf_counter()

WFs_input_folder = "WFs_v7_zcut_noNormalization"
WFs_output_folder = f"WFs_v16_{cfg.IA_model}_may22"

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


c = ISTF.constants['c']

H0 = ISTF.primary['h_0'] * 100
Om0 = ISTF.primary['Om_m0']
Ob0 = ISTF.primary['Om_b0']

gamma = ISTF.extensions['gamma']

z_edges = ISTF.photoz_bins['zbin_edges']
z_m = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']

z_minus = z_edges[:-1]
z_plus = z_edges[1:]
z_0 = z_m / np.sqrt(2)
z_mean = (z_plus + z_minus) / 2
z_min = z_edges[0]
z_max = z_edges[-1]

f_out = ISTF.photoz_pdf['f_out']
sigma_b = ISTF.photoz_pdf['sigma_b']
sigma_o = ISTF.photoz_pdf['sigma_o']
c_b = ISTF.photoz_pdf['c_b']
c_o = ISTF.photoz_pdf['c_o']
z_b = ISTF.photoz_pdf['z_b']
z_o = ISTF.photoz_pdf['z_o']

A_IA = ISTF.IA_free['A_IA']
eta_IA = ISTF.IA_free['eta_IA']
beta_IA = ISTF.IA_free['beta_IA']
C_IA = ISTF.IA_fixed['C_IA']

IA_model = cfg.IA_model
if IA_model == 'eNLA':
    beta_IA = 2.17
elif IA_model == 'zNLA':
    beta_IA = 0.0

simps_z_step_size = 1e-4

n_bar = np.genfromtxt("%s/output/n_bar.txt" % project_path)
lumin_ratio = np.genfromtxt("%s/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt" % project_path)

print('XXXXXX RECHECK Ox0 in cosmolib')
print('XXXXXXXX RECHECK z_mean')


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
# n_i_import = np.genfromtxt("%s/input/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path) # vincenzo (= davide standard, pare)
# n_i_import = np.genfromtxt(path.parent / "common_data/vincenzo/14may/InputNz/niTab-EP10-RB.dat") # vincenzo, more recent (= davide standard, anzi no!!!!)
# n_i_import = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v3_cut/niz_e-19cut.txt") # davide e-20cut
n_i_import = np.load("%s/output/WF/WFs_v2/niz.npy" % project_path)  # davide standard
# n_i_import_2 = np.genfromtxt("%s/output/WF/%s/niz.txt" %(path, WFs_input_folder)) # davide standard with zcutVincenzo

n_i_import = np.load(f'{cfg.niz_path}/{cfg.niz_filename}')

assert n_i_import.shape[1] == zbins, "n_i_import.shape[1] should be == zbins"





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
    return n_i_old(z_prime, i) * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_old(z, i):
    # integrate in z_prime, it must be the first argument
    result = quad(wil_tilde_integrand_old, z, z_max, args=(z, i))
    return result[0]


def wil_tilde_integrand_vec(z_prime, z, i_array):
    """
    vectorized version of wil_tilde_integrand, useful to fill up the computation of the integrand array for the simpson
    integration
    """
    return n_i_new(i_array, z_prime).T * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


# def wil_tilde_new(z, i_array):
#     # version with quad vec, very slow, I don't know why. It is the i_array that is vectorized, because z_prime is integrated over
#     return quad_vec(wil_tilde_integrand_vec, z, z_max, args=(z, i_array))[0]


def wil_noIA_IST(z, i, wil_tilde_array):
    return ((3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlib.r_tilde(z) * wil_tilde_array.T).T


########################################################### IA
# @njit
def W_IA(z_array, i_array):
    result = (H0 / c) * n_i_new(i_array, z_array).T * csmlib.E(z_array)
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
    return Om0 * (1 + z) ** 3 / csmlib.E(z) ** 2


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
def IA_term(z_array, i_array, Dz_array):
    return ((A_IA * C_IA * Om0 * F_IA(z_array)) / Dz_array * W_IA(z_array, i_array)).T


# @njit
def wil_IA_IST(z_array, i_array, wil_tilde_array, Dz_array):
    return wil_noIA_IST(z_array, i_array, wil_tilde_array) - IA_term(z_array, i_array, Dz_array)


###################### wig ###########################
@njit
def b(i):
    return np.sqrt(1 + z_mean[i])


@njit
def b_new(z):
    for i in range(zbins):
        if z_minus[i] <= z < z_plus[i]:
            return b(i)
        if z >= z_plus[-1]:  # max redshift bin
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
def wig_IST(z_array, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    return b(i) * (n_i_new(i_array, z_array).T / n_bar[i]) * H0 * csmlib.E(z_array) / c


# @njit
def wig_multiBinBias_IST_old(z_array, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    # print(b_new(z), z) # debug
    return b_new(z_array) * (n_i_new(i_array, z_array).T / n_bar[i]) * H0 * csmlib.E(z_array) / c


# vectorized version
def wig_multiBinBias_IST(z_array, i_array):  # with n_bar normalisation (anyway, n_bar = 1 more or less)
    result = bz_array * (n_i_new(i_array, z_array) / n_bar[i_array]).T * H0 * csmlib.E(z_array) / c
    return result.T


@njit
def wig_noBias_IST(z_array, i):  # with n_bar normalisation (anyway, n_bar = 1 more or less) ooo
    return (n_i_new(i_array, z_array).T / n_bar[i]) * H0 * csmlib.E(z_array) / c


# def wig_IST(z_array,i): # without n_bar normalisation
#     return b(i) * n_i_new(z_array, i_array).T *H0*csmlib.E(z_array)/c
# xxx I'm already dividing by c!

########################################################################################################################
########################################################################################################################
########################################################################################################################


# using Sylvain's z
# z = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/input/windows_sylvain/nz_source/z.txt")

# TODO add check on i_array, i in niz must be an int, otherwise the function gets interpolated!!
# TODO re-compute and check n_i(z), maybe compute it with scipy.special.erf

if cfg.use_camb:
    # ! new code - just a test with CAMB WF
    ################# CAMB #####################
    import camb
    from camb import model, initialpower

    Om_c0 = ISTF.primary['Om_m0'] - ISTF.primary['Om_b0'] - ISTF.neutrino_params['Omega_nu']
    omch2 = Om_c0 * ISTF.primary['h_0'] ** 2

    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ISTF.primary['Om_bh2'], omch2=omch2,
                       mnu=ISTF.extensions['m_nu'], omk=ISTF.extensions['Om_k0'], tau=ISTF.other_cosmo_params['tau'])

    pars.InitPower.set_params(As=ISTF.other_cosmo_params['A_s'], ns=ISTF.primary['n_s'], r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    # start with one bin
    # wil = camb.sources.SplinedSourceWindow(source_type = 'lensing')

    pars.SourceWindows = [
        GaussianSourceWindow(redshift=0.001, source_type='counts', bias=b(0), sigma=0.04, dlog10Ndm=-0.2),
        GaussianSourceWindow(redshift=0.5, source_type='lensing', sigma=0.07)]

    results = camb.get_results(pars)
    cls = results.get_source_cls_dict()

    # import vincenzo:
    path_vinc = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Cij-NonLin-eNLA_15gen'
    cl_vinc = np.genfromtxt(f'{path_vinc}/CijLL-LCDM-NonLin-eNLA.dat')

    lmax = 2500
    ls = np.arange(2, lmax + 1)
    # for spectrum in ['W1xW1', 'W2xW2', 'W1xW2']:
    for spectrum in ['W1xW1']:
        plt.loglog(ls, cls[spectrum][2: lmax + 1] * 2 * np.pi / (ls * (ls + 1)), label=spectrum)
        plt.plot(cl_vinc[:, 0], cl_vinc[:, 1])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi$')
    plt.legend()

    # ! end new code - just a test with CAMB WF

# COMPUTE KERNELS

# this is the final frid on which the wf are computed
zpoints = 1_000
z_array = np.linspace(z_min, z_max, zpoints)

# this is the z grid used for all the other computations (ie integration)
zpoints_simps = 1_000
z_prime_array = np.linspace(z_min, z_max, zpoints_simps)

print('precomputing arrays')
Dz_array = np.asarray([D(z) for z in z_array])
bz_array = np.asarray([b_new(z) for z in z_array])

# fill simpson integrand
start = time.perf_counter()
integrand = np.zeros((z_prime_array.size, z_array.size, zbins))
for z_idx, z_val in enumerate(z_array):
    # output order of wil_tilde_integrand_vec is: z_prime, i
    integrand[:, z_idx, :] = wil_tilde_integrand_vec(z_prime_array, z_val, i_array).T
print(f'integrand wil_tilde_integrand with for loop filled in: {(time.perf_counter() - start):.2} s')

start = time.perf_counter()
wil_tilde_array = np.zeros((z_array.size, zbins))
for z_idx, z_val in enumerate(z_array):
    # take the closest value to the desired z - less than 0.1% difference with the desired z
    z_prime_idx = np.argmin(np.abs(z_prime_array - z_val))
    wil_tilde_array[z_idx, :] = simpson(integrand[z_prime_idx:, z_idx, :], z_prime_array[z_prime_idx:], axis=0)
print(f'simpson integral done in: {(time.perf_counter() - start):.2} s')

wig_IST_arr = wig_multiBinBias_IST(z_array, i_array)
wil_IA_IST_arr = wil_IA_IST(z_array, i_array, wil_tilde_array, Dz_array)

plt.figure()
for i in range(zbins):
    plt.plot(z_array, wil_IA_IST_arr[:, i], label=f"wil i={i}")
plt.legend()
plt.grid()
plt.show()

plt.figure()
for i in range(zbins):
    plt.plot(z_array, wig_IST_arr[:, i], label=f"wig i={i}")
plt.legend()
plt.grid()
plt.show()

# insert z array values in the 0-th column
wil_IA_IST_arr = np.insert(wil_IA_IST_arr, 0, z_array, axis=1)
wig_IST_arr = np.insert(wig_IST_arr, 0, z_array, axis=1)

np.save(project_path / f'output/WF/{WFs_output_folder}/wil_IA_IST_nz{zpoints}.npy', wil_IA_IST_arr)
np.save(project_path / f'output/WF/{WFs_output_folder}/wig_IST_nz{zpoints}.npy', wig_IST_arr)

print("the script took %.2f seconds to run" % (time.perf_counter() - script_start))
