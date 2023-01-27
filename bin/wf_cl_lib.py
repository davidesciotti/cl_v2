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
lumin_ratio = np.genfromtxt(f"{project_path}/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt")

warnings.warn('RECHECK Ox0 in cosmolib')
warnings.warn('RECHECK z_mean')


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
    """with quad"""
    integrand = lambda z_p, z: n_of_z(z) * pph(z_p, z)
    numerator = quad(integrand, z_minus[i], z_plus[i], args=z)[0]
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])[0]
    return numerator / denominator


def quad_integrand(z_p, z, pph=pph):
    return n_of_z(z) * pph(z_p, z)


def niz_unnormalized_quad(z, zbin_idx, pph=pph):
    """with quad - 0.620401143 s, faster than quadvec..."""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(quad_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, pph))[0]


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


def niz_normalized(z, zbin_idx, pph=pph):
    """this is a wrapper function which normalizes the result.
    The if-else is needed not to compute the normalization for each z, but only once for each zbin_idx
    Note that the niz_unnormalized_quadvec function is not vectorized in z (its 1st argument)
    """
    warnings.warn("this function should be deprecated")
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


# ! test, delete
z_grid = np.linspace(z_min, z_max, 1000)
niz_analytical_arr = np.array([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)]).T
niz_analytical_arr_norm = normalize_niz_simps(niz_analytical_arr.T, z_grid).T
niz_analytical_norm_interp = interp2d(zbin_idx_array, z_grid, niz_analytical_arr_norm, kind="linear")


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
    return niz(zbin_idx_array, z_prime).T * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_new(z):
    # version with quad vec, very slow, I don't know why.
    # It is the zbin_idx_array that is vectorized, because z_prime is integrated over
    return quad_vec(wil_tilde_integrand_vec, z, z_max, args=(z, zbin_idx_array))[0]


def wil_noIA_IST(z, wil_tilde_array):
    return ((3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlib.r_tilde(z) * wil_tilde_array.T).T


########################################################### IA
# @njit
def W_IA(z_grid):
    result = (H0 / c) * niz(zbin_idx_array, z_grid).T * csmlib.E(z_grid)
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


def build_bias_zgrid(z_grid, zbins=zbins):
    bz_values = np.asarray([b_of_z(zbin_idx) for zbin_idx in range(zbins)])
    bias_zgrid = np.array([stepwise_bias(z, bz_values) for z in z_grid])
    return bias_zgrid


def wig_IST(z_grid, which_wf, bias_zgrid=None):
    if bias_zgrid is None:
        bias_zgrid = build_bias_zgrid(z_grid)

    # TODO There is probably room for optimization here, no need to use the callable for niz, just use the array...
    # something like this (but it's already normalized...)
    # result = (niz_analytical_arr_norm / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c

    result = (niz(zbin_idx_array, z_grid) / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c

    if which_wf == 'with_galaxy_bias':
        result = result * bias_zgrid
        return result.T
    elif which_wf == 'without_galaxy_bias':
        return result.T
    elif which_wf == 'galaxy_bias_only':
        return bias_zgrid
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


########################################################################################################################
########################################################################################################################
########################################################################################################################

zpoints = 1000
z_grid = np.linspace(z_min, z_max, zpoints)


# TODO re-compute and check niz_unnorm_quad(z), maybe compute it with scipy.special.erf


def instantiate_PyCCL_cosmology():
    Om_c0 = ISTF.primary['Om_m0'] - ISTF.primary['Om_b0']
    cosmo = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF.primary['Om_b0'], w0=ISTF.primary['w_0'],
                          wa=ISTF.primary['w_a'], h=ISTF.primary['h_0'], sigma8=ISTF.primary['sigma_8'],
                          n_s=ISTF.primary['n_s'], m_nu=ISTF.extensions['m_nu'],
                          Omega_k=1 - (Om_c0 + ISTF.primary['Om_b0']) - ISTF.extensions['Om_Lambda0'])
    return cosmo


def wig_PyCCL(z_grid, which_wf, bias_zgrid=None, cosmo=None, return_PyCCL_object=False):

    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_PyCCL_cosmology()

    # build bias_zgrid
    if bias_zgrid is None:
        bias_zgrid = build_bias_zgrid(z_grid)

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid)

    wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_grid, niz_normalized_arr[zbin_idx, :]),
                                          bias=(z_grid, bias_zgrid), mag_bias=None) for zbin_idx in range(zbins)]

    if return_PyCCL_object:
        return wig


    a_arr = 1 / (1 + z_grid)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    wig_nobias_PyCCL_arr = np.asarray([wig[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    if which_wf == 'with_galaxy_bias':
        result = wig_nobias_PyCCL_arr[:, 0, :] * bias_zgrid
        return result.T
    elif which_wf == 'without_galaxy_bias':
        return wig_nobias_PyCCL_arr[:, 0, :].T
    elif which_wf == 'galaxy_bias_only':
        return bias_zgrid
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


def wil_PyCCL(z_grid, which_wf, cosmo=None, return_PyCCL_object=False):
    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_PyCCL_cosmology()

    # Intrinsic alignment
    IAFILE = lumin_ratio
    z_grid_IA = IAFILE[:, 0]
    growth_factor_PyCCL = ccl.growth_factor(cosmo, a=1 / (1 + z_grid_IA))  # validated against mine
    FIAzNoCosmoNoGrowth = -1 * 1.72 * C_IA * (1 + z_grid_IA) ** eta_IA * IAFILE[:, 1] ** beta_IA
    FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / growth_factor_PyCCL

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid)

    # compute the tracer objects
    wil = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_grid, niz_normalized_arr[zbin_idx, :]),
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


# insert z array values in the 0-th column
# wil_IA_IST_arr = np.insert(wil_IA_IST_arr, 0, z_grid, axis=1)
# wig_IST_arr = np.insert(wig_IST_arr, 0, z_grid, axis=1)

def cl_PyCCL(wf_A, wf_B, ell, zbins, cosmo=None, pk=None):

    # instantiate cosmology
    if cosmo is None:
        cosmo = instantiate_PyCCL_cosmology()

    if pk is None:
        kmin, kmax, nk = 1e-4, 1e1, 500
        k_arr = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        z_grid = np.linspace(0, 4, 500)
        a_arr = 1 / (1 + z_grid)
        pk = ccl.nonlin_matter_power(cosmo, k, a)
        pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

    if np.array_equal(wf_A, wf_B):
        cl = np.array([[ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=pk)
                        for zi in range(zbins)]
                       for zj in range(zi, zbins)])
        cl = cl + cl.T - np.diag(np.diag(cl))
    else:
        cl = np.array([[ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=pk)
                        for zi in range(zbins)]
                       for zj in range(zbins)])
    return cl
