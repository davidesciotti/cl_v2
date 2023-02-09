"""
this is just a copy of PyCCL_test, because it does not give an integration problem for C_LL
"""


import pickle
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.special import erf
import ray

ray.shutdown()
ray.init()

# get project directory and import useful modules
project_path = '/Users/davide/Documents/Lavoro/Programmi/PyCCL_SSC'
project_path_parent = '/Users/davide/Documents/Lavoro/Programmi'

sys.path.append(f'{project_path_parent}/common_data/common_lib')
import my_module as mm

sys.path.append(f'{project_path_parent}/SSC_restructured_v2/bin')
import ell_values as ell_utils

sys.path.append(f'{project_path_parent}/PyCCL_SSC/config')
import PyCCL_config as cfg
import ISTF_fid_params as ISTF_fid
import mpl_cfg

import wf_cl_lib

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


###############################################################################
###############################################################################
###############################################################################

def bias(z, zi):
    zbins = len(zi[0])
    z_minus = zi[0, :]  # lower edge of z bins
    z_plus = zi[1, :]  # upper edge of z bins
    z_mean = (z_minus + z_plus) / 2  # cener of the z bins

    for i in range(zbins):
        if z_minus[i] <= z < z_plus[i]:
            return b(i, z_mean)
        if z > z_plus[-1]:  # max redshift bin
            return b(9, z_mean)


def b(i, z_mean):
    return np.sqrt(1 + z_mean[i])


def compute_SSC_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):
    cov_SSC_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_SSC_timer = time.perf_counter()

    for i in range(zbins):
        start = time.perf_counter()
        for j in range(zbins):
            for k in range(zbins):
                for l in range(zbins):
                    cov_SSC_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_SSC(cosmo, kernel_A[i], kernel_B[j],
                                                                                      ell, tkka,
                                                                                      sigma2_B=None, fsky=f_sky,
                                                                                      cltracer3=kernel_C[k],
                                                                                      cltracer4=kernel_D[l],
                                                                                      ell2=None,
                                                                                      integration_method=integration_method)
        print(f'i-th redshift bins: {i}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'SSC computed in  {(time.perf_counter() - start_SSC_timer):.2f} s')

    return cov_SSC_6D


def compute_cNG_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):
    cov_cNG_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_cNG_timer = time.perf_counter()

    for i in range(zbins):
        start = time.perf_counter()
        for j in range(zbins):
            for k in range(zbins):
                for l in range(zbins):
                    cov_cNG_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_cNG(cosmo, kernel_A[i], kernel_B[j],
                                                                                      ell=ell, tkka=tkka, fsky=f_sky,
                                                                                      cltracer3=kernel_C[k],
                                                                                      cltracer4=kernel_D[l],
                                                                                      ell2=None,
                                                                                      integration_method=integration_method)
        print(f'i-th redshift bins: {i}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'cNG computed in {(time.perf_counter() - start_cNG_timer):.2f} s')

    return cov_cNG_6D


def compute_3x2pt_PyCCL(PyCCL_func, cosmo, probe_wf_dict, ell, tkka, f_sky, integration_method,
                        probe_ordering, probe_combinations_3x2pt):
    # TODO finish this function
    cov_SSC_3x2pt_dict_10D = {}
    for A, B, C, D in probe_combinations_3x2pt:
        print('3x2pt: working on probe combination ', A, B, C, D)
        cov_SSC_3x2pt_dict_10D[A, B, C, D] = PyCCL_func(cosmo,
                                                        probe_wf_dict[A], probe_wf_dict[B],
                                                        probe_wf_dict[C], probe_wf_dict[D], ell, tkka,
                                                        f_sky, integration_method)
        np.save(
            f'{project_path}/output/covariance/cov_PyCCL_{which_NG}_3x2pt_{A}{B}{C}{D}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}.npy',
            cov_SSC_3x2pt_dict_10D[A, B, C, D])

    # TODO test this by loading the cov_SSC_3x2pt_arr_10D from file (and then storing it into a dictionary)
    # symmetrize the matrix:
    LL = probe_ordering[0][0], probe_ordering[0][1]
    GL = probe_ordering[1][0], probe_ordering[1][1]  # ! what if I use LG? check (it should be fine...)
    GG = probe_ordering[2][0], probe_ordering[2][1]
    # note: the addition is only to have a singe tuple of strings, instead of a tuple of 2 tuples
    cov_SSC_3x2pt_dict_10D[GL + LL] = cov_SSC_3x2pt_dict_10D[LL + GL][...]
    cov_SSC_3x2pt_dict_10D[GG + LL] = cov_SSC_3x2pt_dict_10D[LL + GG][...]
    cov_SSC_3x2pt_dict_10D[GG + GL] = cov_SSC_3x2pt_dict_10D[GL + GG][...]

    return cov_SSC_3x2pt_dict_10D


def cl_PyCCL(cosmo, kernel_A, kernel_B, ell, Pk, zbins):
    result = np.array([[ccl.angular_cl(cosmo, kernel_A[iz], kernel_B[jz], ell, p_of_k_a=Pk)
                        for iz in range(zbins)]
                       for jz in range(zbins)])
    result = np.swapaxes(result, 0, 2)
    return result


def compute_ells(nbl: int, ell_min: int, ell_max: int, recipe):
    """
    doesn't output a dictionary (i.e., is single-probe), which is also cleaner
    """
    if recipe == 'ISTF':
        ell_bins = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
        ells = (ell_bins[1:] + ell_bins[:-1]) / 2
        deltas = np.diff(ell_bins)
    elif recipe == 'ISTNL':
        ell_bins = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
        ells = (ell_bins[:-1] + ell_bins[1:]) / 2.
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bins))
    else:
        raise ValueError('recipe must be either "ISTF" or "ISTNL"')

    return ells, deltas


compute_SSC_PyCCL_ray = ray.remote(compute_SSC_PyCCL)
compute_cNG_PyCCL_ray = ray.remote(compute_cNG_PyCCL)
###############################################################################
###############################################################################
###############################################################################

# ! POTENTIAL ISSUES:
# 1. input files (WF, ell, a, pk...)
# 2. halo model recipe
# 3. ordering of the resulting covariance matrix
# * fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX


# ! settings
ell_recipe = cfg.general_cfg['ell_recipe']
probes = cfg.general_cfg['probes']
which_NGs = cfg.general_cfg['which_NGs']
save_covs = cfg.general_cfg['save_covs']
hm_recipe = cfg.general_cfg['hm_recipe']
GL_or_LG = cfg.general_cfg['GL_or_LG']
ell_min = cfg.general_cfg['ell_min']
ell_max = cfg.general_cfg['ell_max']
nbl = cfg.general_cfg['nbl']
zbins = cfg.general_cfg['zbins']
use_ray = cfg.general_cfg['use_ray']  # TODO finish this!
# ! settings

# get number of redshift pairs
zpairs_auto = int((zbins * (zbins + 1)) / 2)
zpairs_cross = zbins ** 2
zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
Om_c0 = ISTF_fid.primary['Om_m0'] - ISTF_fid.primary['Om_b0']
cosmo = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF_fid.primary['Om_b0'], w0=ISTF_fid.primary['w_0'],
                      wa=ISTF_fid.primary['w_a'], h=ISTF_fid.primary['h_0'], sigma8=ISTF_fid.primary['sigma_8'],
                      n_s=ISTF_fid.primary['n_s'], m_nu=ISTF_fid.extensions['m_nu'],
                      Omega_k=1 - (Om_c0 + ISTF_fid.primary['Om_b0']) - ISTF_fid.extensions['Om_Lambda0'])

################################## Define redshift distribution of sources kernels #####################################
zmin, zmax, dz = 0.001, 2.5, 0.001
ztab = np.arange(zmin, zmax, dz)  # ! should it start from 0 instead?

# for CosmoLike
# zmin, zmax, zsteps = 0.001, 4., 10_000
# ztab = np.linspace(zmin, zmax, zsteps)  # ! should it start from 0 instead?

z_median = ISTF_fid.photoz_bins['z_median']

# TODO import these from IST_fid
zbins_edges = np.array([[zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
                        [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]])
# assert (zbins == len(zbins_edges[0])), 'zbins and zbins_edges do not match'

# other useful parameters
n_gal = ISTF_fid.other_survey_specs['n_gal']
survey_area = ISTF_fid.other_survey_specs['survey_area']
f_sky = survey_area * (np.pi / 180) ** 2 / (4 * np.pi)
# n_gal_degsq = n_gal * (180 * 60 / np.pi) ** 2
# sigma_e = ISTF_fid.other_survey_specs['sigma_eps']


fout = ISTF_fid.photoz_pdf['f_out']
cb, zb, sigmab = ISTF_fid.photoz_pdf['c_b'], ISTF_fid.photoz_pdf['z_b'], ISTF_fid.photoz_pdf['sigma_b']
c0, z0, sigma0 = ISTF_fid.photoz_pdf['c_o'], ISTF_fid.photoz_pdf['z_o'], ISTF_fid.photoz_pdf['sigma_o']

nzEuclid = n_gal * (ztab / z_median * np.sqrt(2)) ** 2 * np.exp(-(ztab / z_median * np.sqrt(2)) ** 1.5)

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb * (cb * fout *
                                                    (erf((ztab - z0 - c0 * zbins_edges[0, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigma0) -
                                                     erf((ztab - z0 - c0 * zbins_edges[1, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigma0)) +
                                                    c0 * (1 - fout) *
                                                    (erf((ztab - zb - cb * zbins_edges[0, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigmab) -
                                                     erf((ztab - zb - cb * zbins_edges[1, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigmab))) for iz in range(zbins)])

# normalize nz: this should be the denominator of Eq. (112) of IST:f
for i in range(zbins):
    norm_factor = np.sum(nziEuclid[i, :]) * dz
    nziEuclid[i, :] /= norm_factor

# Intrinsic alignment and galaxy bias
IAFILE = np.genfromtxt(f'{project_path}/input/scaledmeanlum-E2Sa.dat')
FIAzNoCosmoNoGrowth = -1 * 1.72 * 0.0134 * (1 + IAFILE[:, 0]) ** (-0.41) * IAFILE[:, 1] ** 2.17
FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / ccl.growth_factor(cosmo, 1 / (
        1 + IAFILE[:, 0]))

b_array = np.asarray([bias(z, zbins_edges) for z in ztab])

# compute the kernels
wil = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
       for iz in range(zbins)]
wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
                                      mag_bias=None) for iz in range(zbins)]

# Import fiducial P(k,z)
PkFILE = np.genfromtxt(f'{project_path}/input/pkz-Fiducial.txt')

# ! XXX are the units correct?
# Populate vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
zlist = np.unique(PkFILE[:, 0])
k_points = int(len(PkFILE[:, 2]) / len(zlist))
klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
z_points = len(zlist)
Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)  # it's the natural log, not log10
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)


# ! compute cls, just as a test
ells_LL, _ = compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')
cl_LL = cl_PyCCL(cosmo, wil, wil, ells_LL, Pk, zbins)

# ! the problem is in my_wil
# not in the cosmo object
my_wil = wf_cl_lib.wil_PyCCL_obj = wf_cl_lib.wil_PyCCL(ztab, 'with_IA', cosmo='ISTF_fiducial', return_PyCCL_object=True)
cl_LL_v2 = wf_cl_lib.cl_PyCCL(my_wil, my_wil, ells_LL, zbins, is_auto_spectrum=True, pk2d=Pk, cosmo='ISTF_fiducial')

ell_idx, i, j = 0, 0, 0
plt.plot(ells_LL, cl_LL[:, i, j], label='cl_LL')
plt.plot(ells_LL, cl_LL_v2[:, i, j], '--', label='cl_LL_v2')
plt.yscale("log")
plt.legend()
plt.show()

mm.compare_arrays(cl_LL, cl_LL_v2)
