import warnings

import scipy
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf

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
import config_wlcl as cfg

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

# interpolating to speed up
# with z cut following Vincenzo's niz
# with n_bar normalisation
# with "multi-bin" b_of_z
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

z_edges = ISTF.photoz_bins['all_zbin_edges']
z_median = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']
z_minus = ISTF.photoz_bins['z_minus']
z_plus = ISTF.photoz_bins['z_plus']

z_0 = z_median / np.sqrt(2)
z_mean = (z_plus + z_minus) / 2
z_min = z_edges[0]
z_max = 4
sqrt2 = np.sqrt(2)

f_out = ISTF.photoz_pdf['f_out']
c_b, z_b, sigma_b = ISTF.photoz_pdf['c_b'], ISTF.photoz_pdf['z_b'], ISTF.photoz_pdf['sigma_b']
c_o, z_o, sigma_o = ISTF.photoz_pdf['c_o'], ISTF.photoz_pdf['z_o'], ISTF.photoz_pdf['sigma_o']

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

n_bar = np.genfromtxt(f"{project_path}/output/n_bar.txt")
n_gal = ISTF.other_survey_specs['n_gal']
lumin_ratio = np.genfromtxt(f"{project_path}/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt")

warnings.warn('RECHECK Ox0 in cosmolib')
warnings.warn('RECHECK z_mean')
warnings.warn('n_gal prefactor has an effect? Do I normalize the distribution somewhere?')
warnings.warn('Stefanos niz are different from mines! Check the unnormalized ones')


####################################### function definition


@njit
def pph(z, z_p):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_b * (1 + z)) * \
                     np.exp(-0.5 * ((z - c_b * z_p - z_b) / (sigma_b * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_o * (1 + z)) * \
                      np.exp(-0.5 * ((z - c_o * z_p - z_o) / (sigma_o * (1 + z))) ** 2)
    return first_addendum + second_addendum


@njit
def n(z):
    result = n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))
    return result


################################## niz_unnorm_quad(z) ##############################################


# ! load or compute niz_unnorm_quad(z)
if cfg.load_external_niz:
    niz_import = np.genfromtxt(f'{cfg.niz_path}/{cfg.niz_filename}')
    # store and remove the redshift values, ie the 1st column
    z_values_from_nz = niz_import[:, 0]
    niz_import = niz_import[:, 1:]

    assert niz_import.shape[1] == zbins, "niz_import.shape[1] should be == zbins"

    # normalization array
    n_bar = simps(niz_import, z_values_from_nz, axis=0)
    if not np.allclose(n_bar, np.ones(zbins), rtol=0.01, atol=0):
        print('It looks like the input niz_unnorm_quad(z) are not normalized (they differ from 1 by more than 1%)')


def n_i_old(z, i):
    n_i_interp = interp1d(niz_import[:, 0], niz_import[:, i + 1], kind="linear")
    result_array = n_i_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


zbin_idx_array = np.asarray(range(zbins))
assert zbin_idx_array.dtype == 'int64', "zbin_idx_array.dtype should be 'int64'"
niz_import_cpy = niz_import.copy()  # remove redshift column

# note: this is NOT an interpolation in i, which are just the bin indices and will NOT differ from the values 0, 1, 2
# ... 9. The purpose is just to have a 2D vectorized callable.
niz = interp2d(zbin_idx_array, z_values_from_nz, niz_import_cpy, kind="linear")


# note: the normalization of n(z) should be unimportant, here I compute a ratio
# where n(z) is present both at the numerator and denominator!

def n_i(z, i):
    """with quad"""
    integrand = lambda z_p, z: n(z) * pph(z, z_p)
    numerator = quad(integrand, z_minus[i], z_plus[i], args=z)[0]
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])[0]
    return numerator / denominator


def niz_unnormalized_quad(z, zbin_idx, pph=pph):
    """with quad"""
    integrand = lambda z_p, z: n(z) * pph(z, z_p)
    return quad(integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=z)[0]


# equal number of points per bin
zp_points = 500
zp_points_per_bin = int(zp_points / zbins)
zp_bin_grid = np.zeros((zbins, zp_points_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_points_per_bin)

# alternative: equispaced grid with z_edges added (does *not* work well, needs a lot of samples!!)
zp_grid = np.linspace(0, 4, 4000)
zp_grid = np.concatenate((z_edges, zp_grid))
zp_grid = np.unique(zp_grid)
zp_grid = np.sort(zp_grid)
z_edges_idxs = np.array(
    [np.where(zp_grid == z_edges[i])[0][0] for i in range(z_edges.shape[0])])  # indices of z_edges in zp_grid


def niz_unnormalized_simps(z_grid, zbin_idx, pph=pph):
    """numerator of Eq. (112) of ISTF, with simpson integration
    Not too fast (3.0980 s for 500 z_p points)"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'  # TODO check if these slow down the code using scalene
    niz_unnorm_integrand = np.array([pph(zp_bin_grid[zbin_idx, :], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :], axis=1)
    niz_unnorm_integral *= n(z_grid)
    return niz_unnorm_integral


def niz_unnormalized_simps_2(z_arr, zbin_idx, pph=pph):
    """numerator of Eq. (112) of ISTF, with simpson integration and "global" grid"""
    warnings.warn('this function does not work well, needs very high number of samples;'
                  ' the zp_bin_grid sampling is better')
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_minus = z_edges_idxs[zbin_idx]
    z_plus = z_edges_idxs[zbin_idx + 1]
    niz_unnorm_integrand = np.array([pph(z, zp_grid[z_minus:z_plus]) for z in z_arr])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_grid[z_minus:z_plus], axis=1)
    return niz_unnorm_integral * n(z_arr)


def niz_unnormalized_quadvec(z, zbin_idx, pph=pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec
    """
    warnings.warn("niz_unnormalized_quadvec does not seem to work... check and time against simpson")
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    integrand = lambda z_p, z: n(z) * pph(z, z_p)
    niz_unnorm = quad_vec(integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=z)[0]
    return niz_unnorm


def niz_normalization(zbin_idx, niz_unnormalized_func, pph):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_edges[0], z_edges[-1], args=(zbin_idx, pph))[0]


def normalize_niz(niz_unnorm_arr, z_arr):
    """ much more convenient; uses simps, and accepts as input an array of shape (zbins, z_points)"""
    norm_factor = simps(niz_unnorm_arr, z_arr)
    niz_norm = (niz_unnorm_arr.T / norm_factor).T
    return niz_norm


def niz_normalized(z, zbin_idx, pph):
    """this is a wrapper function which normalizes the result.
    The if-else is needed not to compute the normalization for each z, but only once for each zbin_idx
    Note that the niz_unnormalized_quadvec function is not vectorized in z (its 1st argument)
    """
    warnings.warn("this function should be deprecated")
    if type(z) == float or type(z) == int:
        return niz_unnormalized_quadvec(z, zbin_idx, pph) / niz_normalization(zbin_idx, niz_unnormalized_quadvec, pph)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_quadvec(z_value, zbin_idx, pph) for z_value in z])
        return niz_unnormalized_arr / niz_normalization(zbin_idx, niz_unnormalized_quadvec, pph)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


def niz_unnorm_analytical(z, zbin_idx):
    """the one used by Stefano in the PyCCL notebook"""
    addendum_1 = erf((z - z_o - c_o * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_o))
    addendum_2 = erf((z - z_o - c_o * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_o))
    addendum_3 = erf((z - z_b - c_b * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_b))
    addendum_4 = erf((z - z_b - c_b * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_b))

    result = n(z) / (2 * c_o * c_b) * \
             (c_b * f_out * (addendum_1 - addendum_2) + c_o * (1 - f_out) * (addendum_3 - addendum_4))
    return result


z_arr = np.linspace(0, 4, 1000)

# niz_unnormalized_quadvec_arr = np.asarray([niz_unnormalized_quadvec(z_arr, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_simps_2_arr = np.asarray([niz_unnormalized_simps_2(z_arr, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_simps_arr = np.asarray([niz_unnormalized_simps(z_arr, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_quad_arr = np.asarray([[niz_unnormalized_quad(z, zbin_idx)
                                         for z in z_arr]
                                        for zbin_idx in range(zbins)])
niz_unnormalized_analytical = np.asarray([niz_unnorm_analytical(z_arr, zbin_idx) for zbin_idx in range(zbins)])
# niz_unnormalized_dav_2 = niz(np.array(range(10)), z_arr).T

# normalize nz: this should be the denominator of Eq. (112) of IST:f
norm_factor_stef = simps(niz_unnormalized_analytical, z_arr)

# niz_normalized_quadvec_arr = normalize_niz(niz_unnormalized_quadvec_arr, z_arr)
niz_normalized_quad_arr = normalize_niz(niz_unnormalized_quad_arr, z_arr)
niz_normalized_simps_arr = normalize_niz(niz_unnormalized_simps_arr, z_arr)
niz_normalized_simps_2_arr = normalize_niz(niz_unnormalized_simps_2_arr, z_arr)
niz_normalized_analytical = normalize_niz(niz_unnormalized_analytical, z_arr)
niz_normalized_cfp = np.load('/Users/davide/Documents/Lavoro/Programmi/cl_v2/input/niz_cosmicfishpie.npy')

# plot them
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
zbin_idx = 5
ax.plot(z_arr, niz_normalized_analytical[zbin_idx], label='analytical', lw=1.3)
ax.plot(z_arr, niz_normalized_quad_arr[zbin_idx], label='quad', lw=1.3)
ax.plot(z_arr, niz_normalized_simps_arr[zbin_idx], label='simps', lw=1.3)
ax.plot(z_arr, niz_normalized_simps_2_arr[zbin_idx], label='simps_2', lw=1.3)
# ax.plot(z_arr, niz_normalized_cfp[zbin_idx], label='cfp', lw=1.3)
ax.set_xlabel('z')
ax.set_ylabel('n_i(z)')
ax.legend()
plt.show()


################################## end niz ##############################################


# @njit
def wil_tilde_integrand_old(z_prime, z, i):
    return n_i_old(z_prime, i) * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_old(z, i):
    # integrate in z_prime, it must be the first argument
    result = quad(wil_tilde_integrand_old, z, z_max, args=(z, i))
    return result[0]


def wil_tilde_integrand_vec(z_prime, z, zbin_idx_array):
    """
    vectorized version of wil_tilde_integrand, useful to fill up the computation of the integrand array for the simpson
    integration
    """
    return niz(zbin_idx_array, z_prime).T * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


# def wil_tilde_new(z, zbin_idx_array):
#     # version with quad vec, very slow, I don't know why. It is the zbin_idx_array that is vectorized, because z_prime is integrated over
#     return quad_vec(wil_tilde_integrand_vec, z, z_max, args=(z, zbin_idx_array))[0]


def wil_noIA_IST(z, wil_tilde_array):
    return ((3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlib.r_tilde(z) * wil_tilde_array.T).T


########################################################### IA
# @njit
def W_IA(z_array, zbin_idx_array):
    result = (H0 / c) * niz(zbin_idx_array, z_array).T * csmlib.E(z_array)
    return result


# def L_ratio(z):
#     lumin_ratio_interp1d = interp1d(lumin_ratio[:, 0], lumin_ratio[:, 1], kind='linear')
#     result_array = lumin_ratio_interp1d(z)  # z is considered as an array
#     result = result_array.item()  # otherwise it would be a 0d array
#     return result

# test this
lumin_ratio_z_values = lumin_ratio[:, 0]
L_ratio = interp1d(lumin_ratio_z_values, lumin_ratio[:, 1], kind='linear')


# @njit
def F_IA(z):
    result = (1 + z) ** eta_IA * (L_ratio(z)) ** beta_IA
    return result


# use formula 23 of ISTF paper for Om(z)
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
def IA_term(z_grid, zbin_idx_array, Dz_array):
    return ((A_IA * C_IA * Om0 * F_IA(z_grid)) / Dz_array * W_IA(z_grid, zbin_idx_array)).T


# @njit
def wil_IA_IST(z_grid, zbin_idx_array, wil_tilde_array, Dz_array):
    return wil_noIA_IST(z_grid, wil_tilde_array) - IA_term(z_grid, zbin_idx_array, Dz_array)


###################### wig ###########################
def b_of_z(zbin_idx):
    return np.sqrt(1 + z_mean[zbin_idx])


def stepwise_bias(z, bz_values):
    """bz_values is the array containing one bz value per redshift bin; this function copies this value for each z
    in the bin range"""
    for zbin_idx in range(zbins):

        if z < z_minus[zbin_idx]:  # e.g. z = 0 and z_minus[0] = 0.001; in this case, return bias of the first bin
            return bz_values[0]
        if z_minus[zbin_idx] <= z < z_plus[zbin_idx]:
            return bz_values[zbin_idx]
        if z >= z_plus[-1]:  # max redshift bin
            return bz_values[zbins - 1]  # last value


def wig_IST(z_grid, bias_zgrid=None):

    if bias_zgrid is None:
        bz_values = np.asarray([b_of_z(zbin_idx) for zbin_idx in range(zbins)])
        bias_zgrid = np.array([stepwise_bias(z, bz_values) for z in z_grid])

    result = (niz(zbin_idx_array, z_grid) / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c * bias_zgrid
    return result.T


########################################################################################################################
########################################################################################################################
########################################################################################################################

zpoints = 1000
z_arr = np.linspace(z_min, z_max, zpoints)


# using Sylvain's z
# z = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/input/windows_sylvain/nz_source/z.txt")

# TODO re-compute and check niz_unnorm_quad(z), maybe compute it with scipy.special.erf


# ! compute wf

# this is the final grid on which the wf are computed

# this is the z grid used for all the other computations (i.e., integration)

def wil_final(z_grid, which_wf):


    # precompute growth factor
    Dz_array = np.asarray([D(z) for z in z_grid])

    # fill simpson integrand
    zpoints_simps = 700
    z_prime_array = np.linspace(z_min, z_max, zpoints_simps)
    integrand = np.zeros((z_prime_array.size, z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # output order of wil_tilde_integrand_vec is: z_prime, i
        integrand[:, z_idx, :] = wil_tilde_integrand_vec(z_prime_array, z_val, zbin_idx_array).T

    # integrate with simpson to obtain wil_tilde
    wil_tilde_array = np.zeros((z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # take the closest value to the desired z - less than 0.1% difference with the desired z
        z_prime_idx = np.argmin(np.abs(z_prime_array - z_val))
        wil_tilde_array[z_idx, :] = simpson(integrand[z_prime_idx:, z_idx, :], z_prime_array[z_prime_idx:], axis=0)

    if which_wf == 'with_IA':
        return wil_IA_IST(z_grid, zbin_idx_array, wil_tilde_array, Dz_array)
    elif which_wf == 'without_IA':
        return wil_noIA_IST(z_grid, wil_tilde_array)
    elif which_wf == 'IA_only':
        return W_IA(z_grid, zbin_idx_array).T




# ! compute wf with PyCCL
# instantiate cosmology
def wf_PyCCL(cosmo=None):
    if cosmo is None:
        Om_c0 = ISTF.primary['Om_m0'] - ISTF.primary['Om_b0']
        cosmo = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF.primary['Om_b0'], w0=ISTF.primary['w_0'],
                              wa=ISTF.primary['w_a'], h=ISTF.primary['h_0'], sigma8=ISTF.primary['sigma_8'],
                              n_s=ISTF.primary['n_s'], m_nu=ISTF.extensions['m_nu'],
                              Omega_k=1 - (Om_c0 + ISTF.primary['Om_b0']) - ISTF.extensions['Om_Lambda0'])

    # Intrinsic alignment
    # IAFILE = np.genfromtxt(project_path / 'input/scaledmeanlum-E2Sa.dat')
    # warnings.warn('could this cause some difference?')
    IAFILE = lumin_ratio
    z_arr_IA = IAFILE[:, 0]
    FIAzNoCosmoNoGrowth = -1 * 1.72 * C_IA * (1 + IAFILE[:, 0]) ** eta_IA * IAFILE[:, 1] ** beta_IA
    FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / \
           ccl.growth_factor(cosmo, 1 / (1 + IAFILE[:, 0]))

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnorm_analytical(z_arr, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz(niz_unnormalized, z_arr)

    # compute the tracer objects
    wil = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_arr, niz_normalized_arr[zbin_idx, :]), ia_bias=(z_arr_IA, FIAz),
                                         use_A_ia=False) for zbin_idx in range(zbins)]
    wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, niz_normalized_arr[zbin_idx, :]),
                                          bias=(z_arr, bias_zgrid),
                                          mag_bias=None) for zbin_idx in range(zbins)]

    # get the radial kernels
    # comoving distance of z
    a_arr = 1 / (1 + z_arr)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    # get radial kernels
    wil_PyCCL_arr = np.asarray([wil[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])
    wig_nobias_PyCCL_arr = np.asarray([wig[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    # these methods do not return ISTF kernels:
    # for wil, I have the 2 components w_gamma and w_IA separately, see below
    wil_noIA_PyCCL_arr = wil_PyCCL_arr[:, 0, :]
    wil_IAonly_PyCCL_arr = wil_PyCCL_arr[:, 1, :]
    wil_IA_PyCCL_arr = wil_noIA_PyCCL_arr - (A_IA * C_IA * Om0 * F_IA(z_arr)) / Dz_array * wil_IAonly_PyCCL_arr

    # for wig, I have to multiply by bias_zgrid if I want to include bias
    wig_bias_PyCCL_arr = wig_nobias_PyCCL_arr * bias_zgrid

    results = {'wil_noIA_PyCCL_arr': wil_noIA_PyCCL_arr,
               'wil_IAonly_PyCCL_arr': wil_IAonly_PyCCL_arr,
               'wil_IA_PyCCL_arr': wil_IA_PyCCL_arr,
               'wig_bias_PyCCL_arr': wig_bias_PyCCL_arr,
               'wig_nobias_PyCCL_arr': wig_nobias_PyCCL_arr,
               'bias_zgrid': bias_zgrid,
               }
    return results




# insert z array values in the 0-th column
# wil_IA_IST_arr = np.insert(wil_IA_IST_arr, 0, z_arr, axis=1)
# wig_IST_arr = np.insert(wig_IST_arr, 0, z_arr, axis=1)





print("the script took %.2f seconds to run" % (time.perf_counter() - script_start))
