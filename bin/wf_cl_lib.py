import warnings

import scipy
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import type_enforced
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
from functools import partial

# project_path = Path.cwd().parent
project_path = '/Users/davide/Documents/Lavoro/Programmi/cl_v2'
project_path_parent = '/Users/davide/Documents/Lavoro/Programmi/cl_v2'

# general libraries
sys.path.append(f'{project_path_parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path_parent}/common_data/common_config')
import ISTF_fid_params as ISTF
import mpl_cfg

# config files
sys.path.append(f'{project_path}/config')
import config_wlcl as cfg

sys.path.append(f'{project_path_parent}/SSC_restructured_v2/bin')
import ell_values

# project modules
# sys.path.append(f'{project_path}/bin')

# update plot pars
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')

###############################################################################
###############################################################################
###############################################################################


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
z_max = cfg.z_max
sqrt2 = np.sqrt(2)

f_out = ISTF.photoz_pdf['f_out']
c_in, z_in, sigma_in = ISTF.photoz_pdf['c_b'], ISTF.photoz_pdf['z_b'], ISTF.photoz_pdf['sigma_b']
c_out, z_out, sigma_out = ISTF.photoz_pdf['c_o'], ISTF.photoz_pdf['z_o'], ISTF.photoz_pdf['sigma_o']

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
lumin_ratio_file = np.genfromtxt(f"{project_path}/input/scaledmeanlum-E2Sa.dat")

z_grid_lumin_ratio = lumin_ratio_file[:, 0]
lumin_ratio_func = interp1d(z_grid_lumin_ratio, lumin_ratio_file[:, 1], kind='linear', fill_value='extrapolate')

z_max_cl = cfg.z_max_cl
k_grid = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.k_points)
z_grid = np.linspace(z_min, z_max_cl, cfg.zsteps_cl)
use_h_units = cfg.use_h_units

warnings.warn('RECHECK Ox0 in cosmolib')
warnings.warn('RECHECK z_mean')
warnings.warn('RECHECK lumin_ratio_extrapolated')


####################################### function definition


@njit
def pph(z_p, z):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_in * (1 + z)) * \
                     np.exp(-0.5 * ((z - c_in * z_p - z_in) / (sigma_in * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_out * (1 + z)) * \
                      np.exp(-0.5 * ((z - c_out * z_p - z_out) / (sigma_out * (1 + z))) ** 2)
    return first_addendum + second_addendum


@njit
def n_of_z(z):
    return n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))


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


# note: the normalization of n_of_z(z) should be unimportant, here I compute a ratio
# where n_of_z(z) is present both at the numerator and denominator!

def n_i(z, i):
    """with quad. normalized"""
    integrand = lambda z_p, z: n_of_z(z) * pph(z_p, z)
    numerator = quad(integrand, z_minus[i], z_plus[i], args=z)[0]
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])[0]
    return numerator / denominator


def niz_unnormalized_quad(z, zbin_idx, pph=pph):
    """with quad - 0.620401143 s, faster than quadvec..."""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return n_of_z(z) * quad(pph, z_minus[zbin_idx], z_plus[zbin_idx], args=(z))[0]


# SIMPSON WITH DIFFERENT POSSIBLE GRIDS:

# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
# equal number of points per bin
zp_points = 500
zp_points_per_bin = int(zp_points / zbins)
zp_bin_grid = np.zeros((zbins, zp_points_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_points_per_bin)


# more pythonic way of instantiating the same grid
# zp_bin_grid = np.linspace(z_min, z_max, zp_points)
# zp_bin_grid = np.append(zp_bin_grid, z_edges)  # add bin edges
# zp_bin_grid = np.sort(zp_bin_grid)
# zp_bin_grid = np.unique(zp_bin_grid)  # remove duplicates (first and last edges were already included)
# zp_bin_grid = np.tile(zp_bin_grid, (zbins, 1))  # repeat the grid for each bin (in each row)
# for i in range(zbins):  # remove all the points below the bin edge
#     zp_bin_grid[i, :] = np.where(zp_bin_grid[i, :] > z_edges[i], zp_bin_grid[i, :], 0)


def niz_unnormalized_simps(z_grid, zbin_idx, pph=pph):
    """numerator of Eq. (112) of ISTF, with simpson integration
    Not too fast (3.0980 s for 500 z_p points)"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'  # TODO check if these slow down the code using scalene
    niz_unnorm_integrand = np.array([pph(zp_bin_grid[zbin_idx, :], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :], axis=1)
    niz_unnorm_integral *= n_of_z(z_grid)
    return niz_unnorm_integral


# alternative: equispaced grid with z_edges added (does *not* work well, needs a lot of samples!!)
zp_grid = np.linspace(z_min, z_max, 4000)
zp_grid = np.concatenate((z_edges, zp_grid))
zp_grid = np.unique(zp_grid)
zp_grid = np.sort(zp_grid)
# indices of z_edges in zp_grid:
z_edges_idxs = np.array([np.where(zp_grid == z_edges[i])[0][0] for i in range(z_edges.shape[0])])


def niz_unnormalized_simps_fullgrid(z_grid, zbin_idx, pph=pph):
    """numerator of Eq. (112) of ISTF, with simpson integration and "global" grid"""
    warnings.warn('this function needs very high number of samples;'
                  ' the zp_bin_grid sampling should perform better')
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_minus = z_edges_idxs[zbin_idx]
    z_plus = z_edges_idxs[zbin_idx + 1]
    niz_unnorm_integrand = np.array([pph(zp_grid[z_minus:z_plus], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_grid[z_minus:z_plus], axis=1)
    return niz_unnorm_integral * n_of_z(z_grid)


def niz_unnormalized_quadvec(z, zbin_idx, pph=pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec.
    ! the difference is that the integrand can be a vector-valued function (in this case in z_p),
    so it's supposedly faster? -> no, it's slower - 5.5253 s
    """
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    niz_unnorm = quad_vec(quad_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, pph))[0]
    return niz_unnorm


def niz_normalization_quad(niz_unnormalized_func, zbin_idx, pph=pph):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_min, z_max, args=(zbin_idx, pph))[0]


def normalize_niz_simps(niz_unnorm_arr, z_grid):
    """ much more convenient; uses simps, and accepts as input an array of shape (zbins, z_points)"""
    norm_factor = simps(niz_unnorm_arr, z_grid)
    niz_norm = (niz_unnorm_arr.T / norm_factor).T
    return niz_norm


def niz_normalized(z, zbin_idx):
    """this is a wrapper function which normalizes the result.
    The if-else is needed not to compute the normalization for each z, but only once for each zbin_idx
    Note that the niz_unnormalized_quadvec function is not vectorized in z (its 1st argument)
    """
    warnings.warn("this function should be deprecated")
    warnings.warn('or add possibility to choose pph')
    if type(z) == float or type(z) == int:
        return niz_unnormalized_quadvec(z, zbin_idx) / niz_normalization_quad(zbin_idx, niz_unnormalized_quadvec)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_quadvec(z_value, zbin_idx) for z_value in z])
        return niz_unnormalized_arr / niz_normalization_quad(zbin_idx, niz_unnormalized_quadvec)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


def niz_unnormalized_analytical(z, zbin_idx):
    """the one used by Stefano in the PyCCL notebook
    by far the fastest, 0.009592 s"""
    addendum_1 = erf((z - z_out - c_out * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_2 = erf((z - z_out - c_out * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_3 = erf((z - z_in - c_in * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_in))
    addendum_4 = erf((z - z_in - c_in * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_in))

    result = n_of_z(z) / (2 * c_out * c_in) * \
             (c_in * f_out * (addendum_1 - addendum_2) + c_out * (1 - f_out) * (addendum_3 - addendum_4))
    return result


################################## end niz ##############################################


# @njit
def wil_tilde_integrand_old(z_prime, z, i):
    return n_i_old(z_prime, i) * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_old(z, i):
    # integrate in z_prime, it must be the first argument
    result = quad(wil_tilde_integrand_old, z, z_max, args=(z, i))
    return result[0]


def wil_tilde_integrand_vec(z_prime, z):
    """
    vectorized version of wil_tilde_integrand, useful to fill up the computation of the integrand array for the simpson
    integration
    """

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_prime, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_prime)

    # return niz(zbin_idx_array, z_prime).T * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))  # old, with interpolator
    return niz_normalized_arr * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_new(z):
    # version with quad vec, very slow, I don't know why.
    # It is the zbin_idx_array that is vectorized, because z_prime is integrated over
    return quad_vec(wil_tilde_integrand_vec, z, z_max, args=(z, zbin_idx_array))[0]


def wil_noIA_IST(z, wil_tilde_array):
    return ((3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlib.r_tilde(z) * wil_tilde_array.T).T


########################################################### IA
# @njit
def W_IA(z_grid):
    warnings.warn("what about the normalization?")
    warnings.warn("different niz for sources and lenses?")

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid)

    # return (H0 / c) * niz(zbin_idx_array, z_grid).T * csmlib.E(z_grid)  # ! old, with interpolator
    return (H0 / c) * niz_normalized_arr * csmlib.E(z_grid)


# @njit
def F_IA(z):
    result = (1 + z) ** eta_IA * (lumin_ratio_func(z)) ** beta_IA
    return result


# use formula 23 of ISTF paper for Om(z)
# @njit
def Om(z):
    return Om0 * (1 + z) ** 3 / csmlib.E(z) ** 2


# @njit
def growth_factor_integrand(x):
    return Om(x) ** gamma / (1 + x)


def growth_factor(z):
    integral = quad(growth_factor_integrand, 0, z)[0]
    return np.exp(-integral)


# @njit
# def IA_term_old(z, i):
#     return (A_IA * C_IA * Om0 * F_IA(z)) / growth_factor(z) * W_IA(z, i)

# @njit
def IA_term(z_grid, growth_factor_arr):
    """new version, vectorized"""
    return ((A_IA * C_IA * Om0 * F_IA(z_grid)) / growth_factor_arr * W_IA(z_grid)).T


# @njit
def wil_IA_IST(z_grid, wil_tilde_array, growth_factor_arr):
    return wil_noIA_IST(z_grid, wil_tilde_array) - IA_term(z_grid, growth_factor_arr)


def wil_final(z_grid, which_wf):
    # precompute growth factor
    growth_factor_arr = np.asarray([growth_factor(z) for z in z_grid])

    # fill simpson integrand
    zpoints_simps = 700
    z_prime_array = np.linspace(z_min, z_max, zpoints_simps)
    integrand = np.zeros((z_prime_array.size, z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # output order of wil_tilde_integrand_vec is: z_prime, i
        integrand[:, z_idx, :] = wil_tilde_integrand_vec(z_prime_array, z_val).T

    # integrate with simpson to obtain wil_tilde
    wil_tilde_array = np.zeros((z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # take the closest value to the desired z - less than 0.1% difference with the desired z
        z_prime_idx = np.argmin(np.abs(z_prime_array - z_val))
        wil_tilde_array[z_idx, :] = simpson(integrand[z_prime_idx:, z_idx, :], z_prime_array[z_prime_idx:], axis=0)

    if which_wf == 'with_IA':
        return wil_IA_IST(z_grid, wil_tilde_array, growth_factor_arr)
    elif which_wf == 'without_IA':
        return wil_noIA_IST(z_grid, wil_tilde_array)
    elif which_wf == 'IA_only':
        return W_IA(z_grid).T
    else:
        raise ValueError('which_wf must be "with_IA", "without_IA" or "IA_only"')


###################### wig ###########################

def b_of_z(z):
    """simple analytical prescription for the linear galaxy bias:
    b(z) = sqrt(1 + z)
    """
    return np.sqrt(1 + z)


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


def build_bias_zgrid_deprecated(z_grid):
    bz_values = np.asarray([b_of_z(z_mean_val) for z_mean_val in z_mean])
    bias_zgrid = np.array([stepwise_bias(z, bz_values) for z in z_grid])
    return bias_zgrid


def build_galaxy_bias_2d_array(bias_values, z_values, zbins, z_grid, bias_model, plot_bias=False):
    """
    builds a 2d array of shape (len(z_grid), zbins) containing the bias values for each redshift bin. The bias values
    can be given as a function of z, or as a constant value for each redshift bin. Each weight funcion will

    :param bias_values: the values of the bias for each redshift bin
    :param z_values: if a linear interpolation is needed, this is the array of z values for which the bias is given
    :param zbins: number of redshift bins
    :param z_grid: the redshift grid on which the bias is evaluated. in general, it does need to be very fine
    :param bias_model: 'unbiased', 'linint', 'constant' or 'step-wise'
    :param plot_bias: whether to plot the bias values for the different redshift bins
    :return: bias_values: array of shape (len(z_grid), zbins) containing the bias values for each redshift bin.
    """

    assert len(bias_values) == zbins, 'bias_values must be an array of length zbins'

    if bias_model == 'unbiased':
        bias_values = np.ones((len(z_grid), zbins))
    elif bias_model == 'linint':
        galaxy_bias_func = scipy.interpolate.interp1d(z_values, bias_values, kind='linear',
                                                      fill_value=(bias_values[0], bias_values[-1]), bounds_error=False)
        bias_values = galaxy_bias_func(z_grid)
        bias_values = np.repeat(bias_values[:, np.newaxis], zbins, axis=1)
    elif bias_model == 'constant':
        # this is the only case in which the bias is different for each zbin
        bias_values = np.repeat(bias_values[np.newaxis, :], len(z_grid), axis=0)
    elif bias_model == 'step-wise':
        bias_values = np.array([stepwise_bias(z, bias_values) for z in z_grid])
        bias_values = np.repeat(bias_values[:, np.newaxis], zbins, axis=1)

    if plot_bias:
        plt.figure()
        plt.title(f'bias_model {bias_model}')
        for zbin_idx in range(zbins):
            plt.plot(z_grid, bias_values[:, zbin_idx], label=f'zbin {zbin_idx}')
        plt.legend()
        plt.show()

    return bias_values


def build_IA_2d_array(lumin_ratio, z_grid_lumin_ratio, cosmo=None, A_IA=None, eta_IA=None, beta_IA=None, C_IA=None,
                      growth_factor=None, omega_c=None, omega_b=None):
    """
    None is the default value, in which case we use ISTF fiducial values (or the cosmo object)
    :param lumin_ratio:
    :param z_grid_lumin_ratio:
    :param cosmo:
    :param A_IA:
    :param C_IA:
    :param eta_IA:
    :param beta_IA:
    :return:
    """

    if cosmo is None:
        cosmo = instantiate_ISTFfid_PyCCL_cosmo_obj()
    if A_IA is None:
        A_IA = ISTF.IA_free['A_IA']
    if eta_IA is None:
        eta_IA = ISTF.IA_free['eta_IA']
    if beta_IA is None:
        beta_IA = ISTF.IA_free['beta_IA']
    if C_IA is None:
        C_IA = ISTF.IA_fixed['C_IA']
    if growth_factor is None:
        growth_factor = ccl.growth_factor(cosmo, a=1 / (1 + z_grid_lumin_ratio))
    if omega_c is None:
        omega_c = cosmo.cosmo.params.Omega_c
    if omega_b is None:
        omega_b = cosmo.cosmo.params.Omega_b

    assert len(growth_factor) == len(z_grid_lumin_ratio), 'growth_factor must have the same length ' \
                                                          'as z_grid_lumin_ratio (it must be computed in these ' \
                                                          'redshifts!)'

    FIAzNoCosmoNoGrowth = -1 * A_IA * C_IA * (1 + z_grid_lumin_ratio) ** eta_IA * lumin_ratio ** beta_IA
    FIAz = FIAzNoCosmoNoGrowth * (omega_c + omega_b) / growth_factor

    return FIAz


def wig_IST(z_grid, which_wf, zbins=10, gal_bias_2d_array=None, bias_model='step-wise'):
    """
    Computes the photometri Galaxy Clustering kernel, which is equal to the Intrinsic Alignment kernel if the sources
    and lenses distributions are equal. The kernel is computed on a grid of redshifts z_grid, and is a 2d array of
    shape (len(z_grid), zbins). The kernel is computed for each redshift bin, and the bias is assumed to be constant
    :param bias_model:
    :param z_grid:
    :param which_wf:
    :param zbins:
    :param gal_bias_2d_array:
    :return:
    """

    if gal_bias_2d_array is None:
        z_values = ISTF.photoz_bins['z_mean']
        bias_values = np.asarray([b_of_z(z) for z in z_values])
        gal_bias_2d_array = build_galaxy_bias_2d_array(bias_values, z_values, zbins, z_grid, bias_model)

    assert gal_bias_2d_array.shape == (len(z_grid), zbins), 'gal_bias_2d_array must have shape (len(z_grid), zbins)'

    # TODO There is probably room for optimization here, no need to use the callable for niz, just use the array...
    # something like this (but it's already normalized...)
    # result = (niz_analytical_arr_norm / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c

    # result = (niz(zbin_idx_array, z_grid) / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c
    result = W_IA(z_grid).T  # it's the same! unless the sources are different

    if which_wf == 'with_galaxy_bias':
        result *= gal_bias_2d_array
        return result
    elif which_wf == 'without_galaxy_bias':
        return result
    elif which_wf == 'galaxy_bias_only':
        return gal_bias_2d_array
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


########################################################################################################################
########################################################################################################################
########################################################################################################################


# TODO re-compute and check niz_unnorm_quad(z), maybe compute it with scipy.special.erf


def instantiate_ISTFfid_PyCCL_cosmo_obj():
    Om_c0 = ISTF.primary['Om_m0'] - ISTF.primary['Om_b0']

    Omega_k = 1 - (Om_c0 + ISTF.primary['Om_b0']) - ISTF.extensions['Om_Lambda0']
    if np.abs(Omega_k) < 1e-10:
        warnings.warn("Omega_k is very small but not exactly 0, probably due to numerical errors. Setting it to 0")
        Omega_k = 0

    cosmo = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF.primary['Om_b0'], w0=ISTF.primary['w_0'],
                          wa=ISTF.primary['w_a'], h=ISTF.primary['h_0'], sigma8=ISTF.primary['sigma_8'],
                          n_s=ISTF.primary['n_s'], m_nu=ISTF.extensions['m_nu'], Omega_k=Omega_k)
    return cosmo


def wig_PyCCL(z_grid, which_wf, gal_bias_2d_array=None, bias_model='step-wise', cosmo=None, return_PyCCL_object=False):
    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_ISTFfid_PyCCL_cosmo_obj()

    # build bias_zgrid
    if gal_bias_2d_array is None:
        assert zbins == 10, 'zbins must be 10 if bias_zgrid is not provided'
        z_values = ISTF.photoz_bins['z_mean']
        bias_values = np.asarray([b_of_z(z) for z in z_values])
        gal_bias_2d_array = build_galaxy_bias_2d_array(bias_values, z_values, zbins, z_grid, bias_model)

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid).T

    assert gal_bias_2d_array.shape == (len(z_grid), zbins), 'gal_bias_2d_array must have shape as (len(z_grid), zbins)'
    assert niz_normalized_arr.shape == (len(z_grid), zbins), 'gal_bias_2d_array must have shape as (len(z_grid), zbins)'

    wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_grid, niz_normalized_arr[:, zbin_idx]),
                                          bias=(z_grid, gal_bias_2d_array[:, zbin_idx]), mag_bias=None)
           for zbin_idx in range(zbins)]

    if return_PyCCL_object:
        return wig

    a_arr = 1 / (1 + z_grid)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    wig_nobias_PyCCL_arr = np.asarray([wig[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    if which_wf == 'with_galaxy_bias':
        result = wig_nobias_PyCCL_arr[:, 0, :] * gal_bias_2d_array.T
        return result.T
    elif which_wf == 'without_galaxy_bias':
        return wig_nobias_PyCCL_arr[:, 0, :].T
    elif which_wf == 'galaxy_bias_only':
        return gal_bias_2d_array
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


def wil_PyCCL(z_grid, which_wf, cosmo=None, return_PyCCL_object=False):
    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_ISTFfid_PyCCL_cosmo_obj()

    # Intrinsic alignment
    z_grid_IA = lumin_ratio_file[:, 0]
    lumin_ratio = lumin_ratio_file[:, 1]
    growth_factor_PyCCL = ccl.growth_factor(cosmo, a=1 / (1 + z_grid_IA))  # validated against mine
    FIAzNoCosmoNoGrowth = -1 * A_IA * C_IA * (1 + z_grid_IA) ** eta_IA * lumin_ratio ** beta_IA
    FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / growth_factor_PyCCL

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid).T

    # compute the tracer objects
    wil = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_grid, niz_normalized_arr[:, zbin_idx]),
                                         ia_bias=(z_grid_IA, FIAz), use_A_ia=False) for zbin_idx in range(zbins)]

    if return_PyCCL_object:
        return wil

    # get the radial kernels
    # comoving distance of z
    a_arr = 1 / (1 + z_grid)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    wil_PyCCL_arr = np.asarray([wil[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    # these methods do not return ISTF kernels:
    # for wil, I have the 2 components w_gamma and w_IA separately, see below

    if which_wf == 'with_IA':
        wil_noIA_PyCCL_arr = wil_PyCCL_arr[:, 0, :]
        wil_IAonly_PyCCL_arr = wil_PyCCL_arr[:, 1, :]
        growth_factor_PyCCL = ccl.growth_factor(cosmo, a=1 / (1 + z_grid))
        result = wil_noIA_PyCCL_arr - (A_IA * C_IA * Om0 * F_IA(z_grid)) / growth_factor_PyCCL * wil_IAonly_PyCCL_arr
        return result.T
    elif which_wf == 'without_IA':
        return wil_PyCCL_arr[:, 0, :].T
    elif which_wf == 'IA_only':
        return wil_PyCCL_arr[:, 1, :].T
    else:
        raise ValueError('which_wf must be "with_IA", "without_IA" or "IA_only"')


################################################# cl_quad computation #######################################################

# TODO these contain cosmology dependence...
cosmo_classy = csmlib.cosmo_classy
cosmo_astropy = csmlib.cosmo_astropy
pk = csmlib.calculate_power(k_grid, z_grid, cosmo_classy, use_h_units=use_h_units)
pk_interp_func = interp2d(k_grid, z_grid, pk)

# wrapper functions, just to shorten the names
pk_nonlin_wrap = partial(csmlib.calculate_power, cosmo_classy=cosmo_classy, use_h_units=use_h_units,
                         Pk_kind='nonlinear')
kl_wrap = partial(csmlib.k_limber, use_h_units=use_h_units, cosmo_astropy=cosmo_astropy)


# these are no longer needed, since I can use partial
# def pk_wrap(k_ell, z, cosmo_classy=cosmo_classy, use_h_units=use_h_units, Pk_kind='nonlinear'):
#     """just a wrapper function to set some args to default values"""
#     return csmlib.calculate_power(k_ell, z, cosmo_classy, use_h_units=use_h_units, Pk_kind=Pk_kind)
#
#
# def kl_wrap(ell, z, use_h_units=use_h_units):
#     """another simple wrapper function, so as not to have to rewrite use_h_units=use_h_units"""
#     return csmlib.k_limber(ell, z, use_h_units=use_h_units)


@type_enforced.Enforcer
def K_ij(z, wf_A, wf_B, i: int, j: int):
    return wf_A(z, j) * wf_B(z, i) / (csmlib.E(z) * csmlib.r(z) ** 2)


def cl_partial_integrand(z, wf_A, wf_B, i: int, j: int, ell):
    return K_ij(z, wf_A, wf_B, i, j) * pk_nonlin_wrap(kl_wrap(ell, z), z)


@type_enforced.Enforcer
def cl_partial_integral(wf_A, wf_B, i: int, j: int, zbin: int, ell):
    result = c / H0 * quad(cl_partial_integrand, z_minus[zbin], z_plus[zbin], args=(wf_A, wf_B, i, j, ell))[0]
    return result


print('THIS BIAS IS WRONG; MOREOVER, AM I NOT INCLUDING IT IN THE KERNELS?')


# summing the partial integrals
@type_enforced.Enforcer
def sum_cl_partial_integral(wf_A, wf_B, i: int, j: int, ell):
    warnings.warn('in this version the bias is not included in the kernels')
    warnings.warn('understand the bias array')
    result = 0
    for zbin in range(zbins):
        result += cl_partial_integral(wf_A, wf_B, i, j, zbin, ell) * bias_array[zbin]
    return result


###### OLD BIAS ##################
def cl_integrand(z, wf_A, wf_B, zi, zj, ell):
    return ((wf_A(z)[zi] * wf_B(z)[zj]) / (csmlib.E(z) * csmlib.r(z) ** 2)) * pk_nonlin_wrap(kl_wrap(ell, z), z)


def cl_quad(wf_A, wf_B, ell, zi, zj):
    """ when used with LG or GG, this implements the "old bias"
    """
    result = c / H0 * quad(cl_integrand, z_min, z_max_cl, args=(wf_A, wf_B, zi, zj, ell))[0]
    # xxx maybe you can try with scipy.integrate.romberg?
    return result


def cl_simps(wf_A, wf_B, ell, zi, zj):
    """ when used with LG or GG, this implements the "old bias"
    """
    integrand = [cl_integrand(z, wf_A, wf_B, zi, zj, ell) for z in z_grid]
    return scipy.integrate.simps(integrand, z_grid)


def get_cl_3D_array(wf_A, wf_B, ell_values, is_auto_spectrum):
    nbl = len(ell_values)
    cl_3D = np.zeros((nbl, zbins, zbins))

    if is_auto_spectrum:
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zi, zbins):
                    cl_3D[ell_idx, zi, zj] = cl_quad(wf_A, wf_B, ell_val, zi, zj)
        cl_3D = mm.symmetrize_2d_array(cl_3D)
    else:
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zbins):
                    cl_3D[ell_idx, zi, zj] = cl_quad(wf_A, wf_B, ell_val, zi, zj)

    return cl_3D


def cl_PyCCL(wf_A, wf_B, ell, zbins, is_auto_spectrum, pk2d, cosmo=None, limber_integration_method='qag_quad'):
    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_ISTFfid_PyCCL_cosmo_obj()

    nbl = len(ell)

    # if pk2d is None:
    #     # TODO implement the computation of the power spectrum here
    #     raise NotImplementedError('pk2d must be provided, not yet implemented to compute it here')
    #     kmin, kmax, nk = 1e-4, 1e1, 500
    #     k_arr = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    #     lk_arr = np.log(k_arr)
    #     z_grid = np.linspace(0, 4, 500)
    #     a_arr = 1 / (1 + z_grid)
    #     # pkfunc = lambda k, a: ccl.nonlin_matter_power(cosmo, k, a)
    #     # this is because the pkfunc signature in ccl.Pk2 must be (k, a), not (cosmo, k, a) 👇
    #     pkfunc = partial(ccl.nonlin_matter_power, cosmo=cosmo)
    #     pk2d = ccl.Pk2D(pkfunc=pkfunc, a_arr=a_arr, lk_arr=lk_arr)

    if is_auto_spectrum:
        cl_3D = np.zeros((nbl, zbins, zbins))
        for zi, zj in zip(np.triu_indices(zbins)[0], np.triu_indices(zbins)[1]):
            cl_3D[:, zi, zj] = ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=pk2d,
                                              limber_integration_method=limber_integration_method)
        for ell in range(nbl):
            cl_3D[ell, :, :] = mm.symmetrize_2d_array(cl_3D[ell, :, :])
    else:
        cl_3D = np.array([[ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=pk2d,
                                          limber_integration_method=limber_integration_method)
                           for zi in range(zbins)]
                          for zj in range(zbins)]).transpose(2, 0, 1)  # transpose to have ell as first axis
    return cl_3D


def stem(Cl_arr, variations_arr, zbins, nbl):
    # instantiate array of derivatives
    dCLL_arr = np.zeros((zbins, zbins, nbl))

    # create copy of the "x" and "y" arrays, because their items could get popped by the stem algorithm
    Cl_arr_cpy = Cl_arr.copy()
    variations_arr_cpy = variations_arr.copy()

    # TODO is there a way to specify the axis along which to fit, instead of having to loop over i, j, ell?
    for i in range(zbins):
        for j in range(zbins):
            for ell in range(nbl):

                # perform linear fit
                angular_coefficient, c = np.polyfit(variations_arr_cpy, Cl_arr_cpy[:, i, j, ell], deg=1)
                fitted_y_values = angular_coefficient * variations_arr_cpy + c

                # check % difference
                perc_diffs = mm.percent_diff(Cl_arr_cpy[:, i, j, ell], fitted_y_values)

                # as long as any element has a percent deviation greater than 1%, remove first and last values
                while np.any(perc_diffs > 1):
                    print('removing first and last values, perc_diffs array:', perc_diffs)

                    # the condition is satisfied, removing the first and last values
                    Cl_arr_cpy = np.delete(Cl_arr_cpy, [0, -1], axis=0)
                    variations_arr_cpy = np.delete(variations_arr_cpy, [0, -1])

                    # re-compute the fit on the reduced set
                    angular_coefficient, intercept = np.polyfit(variations_arr_cpy, Cl_arr_cpy[:, i, j, ell], deg=1)
                    fitted_y_values = angular_coefficient * variations_arr_cpy + intercept

                    # test again
                    perc_diffs = mm.percent_diff(Cl_arr_cpy[:, i, j, ell], fitted_y_values)

                    # plt.figure()
                    # plt.plot(Omega_c_values_toder, fitted_y_values, '--', lw=2, c=colors[iteration])
                    # plt.plot(Omega_c_values_toder, CLL_toder[:, i, j, ell], marker='o', c=colors[iteration])

                # store the value of the derivative
                dCLL_arr[i, j, ell] = angular_coefficient

    return dCLL_arr


def compute_derivatives(fiducial_params, free_params, fixed_params, z_grid, zbins, ell_LL, ell_GG, Pk=None):
    """
    Compute the derivatives of the power spectrum with respect to the free parameters
    """
    # TODO cleanup the function, + make it single-probe

    nbl_WL = len(ell_LL)
    nbl_GC = len(ell_GG)

    percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
                              0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100
    num_variations = len(percentages)

    # dictionary storing the perturbed values of the parameters around the fiducials
    variations = {}
    for key in free_params.keys():
        variations[key] = free_params[key] * (1 + percentages)

    # wa = 0, so the deviations are the percentages themselves
    variations['wa'] = percentages

    # declare cl_quad and dcl vectors
    cl_LL, cl_GL, cl_GG = {}, {}, {}
    dcl_LL, dcl_GL, dcl_GG = {}, {}, {}

    # loop over the free parameters and store the cls in a dictionary
    for free_param_name in free_params.keys():

        # instantiate derivatives array for the given free parameter key
        cl_LL[free_param_name] = np.zeros((num_variations, nbl_WL, zbins, zbins))
        cl_GL[free_param_name] = np.zeros((num_variations, nbl_GC, zbins, zbins))
        cl_GG[free_param_name] = np.zeros((num_variations, nbl_GC, zbins, zbins))

        # loop over the perturbed parameter's (i.e. free_param_name) values, stored in variations[free_param_name]
        for variation_idx, free_params[free_param_name] in enumerate(variations[free_param_name]):
            t0 = time.perf_counter()

            # TODO check if the variations are consistent with the parameter's relations (eg omega_lambda?)
            cosmo = ccl.Cosmology(Omega_c=(free_params['Om'] - free_params['Ob']),
                                  Omega_b=free_params['Ob'],
                                  w0=free_params['wz'],
                                  wa=free_params['wa'],
                                  h=free_params['h'],
                                  sigma8=free_params['s8'],
                                  n_s=free_params['ns'],
                                  m_nu=fixed_params['m_nu'],
                                  Omega_k=0.,
                                  extra_parameters={"camb": {"dark_energy_model": "DarkEnergyPPF"}}  # to cross w = -1
                                  )

            wil_PyCCL_obj = wil_PyCCL(z_grid, 'with_IA', cosmo=cosmo, return_PyCCL_object=True)
            wig_PyCCL_obj = wig_PyCCL(z_grid, 'with_galaxy_bias', cosmo=cosmo, return_PyCCL_object=True)

            cl_LL[free_param_name][variation_idx, :, :, :] = cl_PyCCL(wil_PyCCL_obj, wil_PyCCL_obj, ell_LL, zbins,
                                                                      is_auto_spectrum=True, pk2d=Pk)
            cl_GL[free_param_name][variation_idx, :, :, :] = cl_PyCCL(wig_PyCCL_obj, wil_PyCCL_obj, ell_GG, zbins,
                                                                      is_auto_spectrum=False, pk2d=Pk)
            cl_GG[free_param_name][variation_idx, :, :, :] = cl_PyCCL(wig_PyCCL_obj, wig_PyCCL_obj, ell_GG, zbins,
                                                                      is_auto_spectrum=True, pk2d=Pk)

            # # Computes the WL (w/ and w/o IAs) and GCph kernels
            # A_IA, eta_IA, beta_IA = free_params['Aia'], free_params['eIA'], free_params['bIA']
            # FIAzNoCosmoNoGrowth = - A_IA * CIA * (1 + IAFILE[:, 0]) ** eta_IA * IAFILE[:, 1] ** beta_IA
            #
            # FIAz = FIAzNoCosmoNoGrowth * \
            #        (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / \
            #        ccl.growth_factor(cosmo, 1 / (1 + IAFILE[:, 0]))
            #
            # wil = [
            #     ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
            #     for iz in range(zbins)]
            #
            # wig = [
            #     ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
            #                                    mag_bias=None) for iz in range(zbins)]
            #
            # you can also get the kernel in this way:
            # wil_test = ccl.tracers.get_lensing_kernel(cosmo, dndz=(ztab, nziEuclid[0]), mag_bias=None)
            # a_test = ccl.scale_factor_of_chi(cosmo, wil_test[0])
            # z_test = 1 / a_test - 1

            # Import fiducial P(k,z)
            # PkFILE = np.genfromtxt(project_path / 'input/pkz-Fiducial.txt')
            #
            # # Populates vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
            # zlist = np.unique(PkFILE[:, 0])
            # k_points = int(len(PkFILE[:, 2]) / len(zlist))
            # klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
            # z_points = len(zlist)
            # Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3
            #
            # # Create a Pk2D object
            # a_arr = 1 / (1 + zlist[::-1])
            # lk_arr = np.log(klist)  # it's the natural log, not log10
            # pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

            # the key specifies the parameter, but I still need an array of values - corresponding to the 15 variations over
            # the fiducial values
            # CLL[free_param_name][variation_idx, :, :, :] = np.array([[ccl.angular_cl(cosmo, wil[iz], wil[jz],
            #                                                                          ell, p_of_k_a=None)
            #                                                           for iz in range(zbins)]
            #                                                          for jz in range(zbins)])

            print(
                f'{free_param_name} = {free_params[free_param_name]:.4f} Cls computed in {(time.perf_counter() - t0):.2f} '
                f'seconds')

        # once finished looping over the variations, reset the parameter to its fiducial value
        free_params[free_param_name] = fiducial_params[free_param_name]

        # save the Cls
        dcl_LL[free_param_name] = stem(cl_LL[free_param_name], variations[free_param_name], zbins, nbl_WL)
        dcl_GL[free_param_name] = stem(cl_GL[free_param_name], variations[free_param_name], zbins, nbl_GC)
        dcl_GG[free_param_name] = stem(cl_GG[free_param_name], variations[free_param_name], zbins, nbl_GC)

        print(f'SteM derivative computed for {free_param_name}')
        return dcl_LL, dcl_GL, dcl_GG
