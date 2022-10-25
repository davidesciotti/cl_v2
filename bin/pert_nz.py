import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quadpy
from mpire import WorkerPool
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
from scipy import stats

project_path = Path.cwd().parent

sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))

# project modules
# import proj_lib.cosmo_lib as csmlb
# import config.config as cfg
# general configuration modules
import ISTF_fid_params as ISTF
import mpl_cfg as mpl_cfg

# update plot paramseters
rcParams = mpl_cfg.mpl_rcParams_dict
plt.rcParams.update(rcParams)
matplotlib.use('Qt5Agg')

###############################################################################
###############################################################################
###############################################################################


script_start = time.perf_counter()
c = ISTF.constants['c']

H0 = ISTF.primary['h_0'] * 100
Om0 = ISTF.primary['Om_m0']
Ob0 = ISTF.primary['Om_b0']

gamma = ISTF.extensions['gamma']

z_edges = ISTF.photoz_bins['zbin_edges']
z_minus = z_edges[:-1]
z_plus = z_edges[1:]
z_m = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']

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

n_gal = ISTF.other_survey_specs['n_gal']

A_IA = ISTF.IA_free['A_IA']
eta_IA = ISTF.IA_free['eta_IA']
beta_IA = ISTF.IA_free['beta_IA']
C_IA = ISTF.IA_fixed['C_IA']

simps_z_step_size = 1e-4
sqrt2 = np.sqrt(2)
sqrt2pi = np.sqrt(2 * np.pi)

num_gaussians = 20


# n_bar = np.genfromtxt("%s/output/n_bar.txt" % project_path)
# lumin_ratio = np.genfromtxt("%s/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt" % project_path)


####################################### function definition


@njit
def pph_old(z_p, z):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_b * (1 + z)) * \
                     np.exp(-0.5 * ((z - c_b * z_p - z_b) / (sigma_b * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_o * (1 + z)) * \
                      np.exp(-0.5 * ((z - c_o * z_p - z_o) / (sigma_o * (1 + z))) ** 2)
    return first_addendum + second_addendum


@njit
def base_gaussian(z, z_p, nu_n, c_n, z_n, sigma_n):
    """one of the terms used int the sum of gaussians"""
    result = (nu_n * c_n) / (sqrt2pi * sigma_n * (1 + z)) * np.exp(
        -0.5 * ((z - c_n * z_p - z_n) / (sigma_n * (1 + z))) ** 2)
    return result


def base_gaussian_scipy(z, z_p, nu_n, c_n, z_n, sigma_n):
    """this is much slower than base_gaussian!"""
    return (nu_n * c_n) * stats.norm.pdf(z, loc=c_n * z_p - z_n, scale=sigma_n * (1 + z))


@njit
def pph(z_p, z):
    """nu_n is hust a weight for the sum of gaussians, in this case it's just
     (1 - f_out) * pph_in + f_out * pph_out"""
    result = (1 - f_out) * base_gaussian(z, z_p, nu_n=1, c_n=c_b, z_n=z_b, sigma_n=sigma_b) + \
             f_out * base_gaussian(z, z_p, nu_n=1, c_n=c_o, z_n=z_o, sigma_n=sigma_o)
    return result


def pph_pert(z_p, z, seed, num_gaussians=num_gaussians):
    rng = np.random.default_rng(seed)
    nu_n_arr = rng.uniform(-1, 1, size=num_gaussians)
    c_n = 1

    result = 0
    for i in range(num_gaussians):
        result += base_gaussian(z, z_p, nu_n=nu_n_arr[i], c_n=c_n, z_n=z_n_arr[i], sigma_n=sigma_n_arr[i])


@njit
def n(z):
    result = n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** 1.5)
    # TODO normalize the distribution or not?
    # result = result*(30/0.4242640687118783) # normalising the distribution?
    return result


################################## niz ##############################################

# choose the cut XXX
# n_i_import = np.genfromtxt("%s/input/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path) # vincenzo (= davide standard, pare)
# n_i_import = np.genfromtxt(path.parent / "common_data/vincenzo/14may/InputNz/niTab-EP10-RB.dat") # vincenzo, more recent (= davide standard, anzi no!!!!)
# n_i_import = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v3_cut/niz_e-19cut.txt") # davide e-20cut
# n_i_import = np.load("%s/output/WF/WFs_v2/niz.npy" % project_path)  # davide standard


# n_i_import_2 = np.genfromtxt("%s/output/WF/%s/niz.txt" %(path, WFs_input_folder)) # davide standard with zcutVincenzo


# def n_i_old(z, i):
#     n_i_interp = interp1d(n_i_import[:, 0], n_i_import[:, i + 1], kind="linear")
#     result_array = n_i_interp(z)  # z is considered as an array
#     result = result_array.item()  # otherwise it would be a 0d array
#     return result
#
#
# z_values_from_nz = n_i_import[:, 0]
# i_array = np.asarray(range(zbins))
# n_i_import_cpy = n_i_import.copy()[:, 1:]  # remove redshift column

# note: this is NOT an interpolation in i, which are just the bin indices and will NOT differ from the values 0, 1, 2
# ... 9. The purpose is just to have a 2D vectorized callable.
# n_i_new = interp2d(i_array, z_values_from_nz, n_i_import_cpy, kind="linear")

# note: the normalization of n(z) should be unimportant, here I compute a ratio
# where n(z) is present both at the numerator and denominator!
# as a function, including (of not) the ie-20 cut
# def n_i(z, i):
#     integrand = lambda z_p, z: n(z) * pph(z_p, z)
#     numerator = quad(integrand, z_minus[i], z_plus[i], args=(z))
#     denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])
#     #    denominator = nquad(integrand, [[z_minus[i], z_plus[i]],[z_min, z_max]] )
#     #    return numerator[0]/denominator[0]*3 to have quad(n_i, 0, np.inf = nbar_b/20 = 3)
#     result = numerator[0] / denominator[0]
#     # if result < 6e-19: # remove these 2 lines if you don't want the cut
#     #     result = 0
#     return result


# compute n_i(z) with simpson


# define a grid passing through all the z_edges points, to have exact integration limits

"""
# 500 pts seems to be enough - probably more than what quad uses!
z_grid_norm = np.linspace(z_edges[0], z_edges[-1], 500)
def niz_normalization_simps(i):
    assert type(i) == int, 'i must be an integer'
    integrand = np.asarray([niz_unnorm(z, i) for z in z_grid_norm])
    return simps(integrand, z_grid_norm)
"""

# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
zp_num = 2_000
zp_num_per_bin = int(zp_num / zbins)
zp_grid = np.empty(0)
zp_bin_grid = np.zeros((zbins, zp_num_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_num_per_bin)


def niz_unnorm_simps(z, i):
    """numerator of Eq. (112) of ISTF, with simpson integration"""
    assert type(i) == int, 'i must be an integer'
    niz_unnorm_integrand = pph(zp_bin_grid[i, :], z)
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[i, :])
    niz_unnorm_integral *= n(z)
    return niz_unnorm_integral


def niz_unnormalized(z, i):
    """
    :param z: float, does not accept an array
    """
    assert type(i) == int, 'i must be an integer'
    niz_unnorm = quad_vec(pph, z_edges[i], z_edges[i + 1], args=z)[0]
    niz_unnorm *= n(z)
    return niz_unnorm


def niz_normalization(i, niz_unnormalized_func):
    assert type(i) == int, 'i must be an integer'
    return quad(niz_unnormalized_func, z_edges[0], z_edges[-1], args=i)[0]


def niz_unnorm_stef(z, i):
    """the one used in the PyCCL notebook"""
    addendum_1 = erf((z - z_o - c_o * z_edges[i]) / sqrt2 / (1 + z) / sigma_o)
    addendum_2 = erf((z - z_o - c_o * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_o)
    addendum_3 = erf((z - z_b - c_b * z_edges[i]) / sqrt2 / (1 + z) / sigma_b)
    addendum_4 = erf((z - z_b - c_b * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_b)

    result = n(z) * 1 / 2 / c_o / c_b * \
             (c_b * f_out * (addendum_1 - addendum_2) + c_o * (1 - f_out) * (addendum_3 - addendum_4))
    return result


def niz_norm(z, zbin_idx):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return niz_unnormalized(z, zbin_idx) / niz_normalization_arr[zbin_idx]


########################################################################################################################

niz_normalization_arr = np.asarray([niz_normalization(i, niz_unnormalized) for i in range(10)])

z_num = 2000
z_grid = np.linspace(z_min, z_max, z_num)

# compute normalized and unnormalized n(z)
niz_normalized_array = np.asarray([[niz_norm(z, zbin) for zbin in range(10)] for z in z_grid])
niz_unnormalized_array = np.asarray([[niz_unnormalized(z, zbin) for zbin in range(10)] for z in z_grid])

# insert z array
niz_normalized_array = np.insert(niz_normalized_array, 0, z_grid, axis=1)
niz_unnormalized_array = np.insert(niz_unnormalized_array, 0, z_grid, axis=1)

np.savetxt(f'{project_path}/output/niz/niz_normalized_nz{z_num}.txt', niz_normalized_array, header=f'z_values, n_1(z), n_2(z), ...')
np.savetxt(f'{project_path}/output/niz/niz_unnormalized_nz{z_num}.txt', niz_unnormalized_array, header=f'z_values, n_1(z), n_2(z), ...')

zbin = 1
plt.plot(niz_unnormalized_array[:, 0], niz_unnormalized_array[:, zbin+1], label='niz_unnormalized_list')
plt.plot(niz_normalized_array[:, 0], niz_normalized_array[:, zbin+1], label='niz_normalized_list')

print('done')
