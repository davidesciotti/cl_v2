import matplotlib
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import matplotlib
import time
from pathlib import Path
from numba import njit
import sys
import type_enforced

# get project directory
project_path = Path.cwd().parent
sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))
sys.path.append(str(project_path.parent / 'SSC_restructured_v2/bin'))
sys.path.append(str(project_path.parent / 'SSC_restructured_v2/lib'))

# from SSC_restructured_v2
import my_module as mm
import ell_values_running as ell_utils

# general configuration modules
import ISTF_fid_params as ISTF
import mpl_cfg as mpl_cfg

# this project's modules
import config.config as cfg
import proj_lib.cosmo_lib as csmlb

# update plot pars
rcParams = mpl_cfg.mpl_rcParams_dict
plt.rcParams.update(rcParams)
matplotlib.use('Qt5Agg')

script_start = time.perf_counter()

########################################################################################################################
########################################################################################################################
########################################################################################################################

print('XXXXXX RECHECK Ox0 in cosmolib')
print('XXXXXXXX RECHECK z_mean array (not z_median!)')

# TODO redefine integrals, optimize with simps or stuff
# TODO import with numpy.interp1d, valid for jitted functios?
# TODO check k and z arrays
# TODO fix z_array -> fatto? (this is an old TODO)
# TODO check z_max = 2.5 or =4?:  # XXXX see p.25 of IST paper
# TODO rerererererecheck vincenzo's WF and normalize name/shapes of everyone's WF
# TODO check compute_cl_v2, nicer (and faster, in theory)
# TODO check if you can lower z_min
# TODO use Vincenzo's Pk (or TakaBird in general)
# 🐛 could ell values be in log scale? I really don't think so

# set fiducial values and constants
c = ISTF.constants['c']
H0 = ISTF.primary['h_0'] * 100

z_edges = ISTF.photoz_bins['zbin_edges']
z_m = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']

z_edges[0] = 0.03

z_minus = z_edges[:-1]
z_plus = z_edges[1:]
z_mean = (z_plus + z_minus) / 2
z_min = z_edges[0]
z_max = z_edges[-1]

print(f'Warning: z_edges[0] has been set to {z_edges[0]} to make it possible to compute k_limber at high ells and low z')

# configurations
nbl = cfg.nbl
units = cfg.units
z_max_cl = cfg.z_max_cl

if units == 'h/Mpc':
    use_h_units = True
elif units == '1/Mpc':
    use_h_units = False
else:
    raise ValueError('units must be h/Mpc or 1/Mpc')

if not cfg.useIA:
    raise ValueError('cfg.useIA must True for the moment')

# TODO this naming is not the best: use h units means that I use or want them?
if units == '1/Mpc':
    use_h_units = False
elif units == 'h/Mpc':
    use_h_units = True
else:
    raise ValueError('units must be 1/Mpc or h/Mpc')

path_WF = project_path.parent / f"common_data/everyones_WF_from_Gdrive/{cfg.whos_wf}"
if cfg.whos_wf == 'vincenzo':
    raise ValueError('vincenzos WF seem to have some problem!')
    wil_import = np.genfromtxt(f"{path_WF}/wil_vincenzo_{IA_flag}_IST_nz{cfg.nz_WF_import}.dat")
    wig_import = np.genfromtxt(f"{path_WF}/wig_vincenzo_IST_nz{cfg.nz_WF_import}.dat")
if cfg.whos_wf == 'marco':
    wil_import = np.load(f"{path_WF}/wil_mar_bia{ISTF.IA_free['beta_IA']}_IST_nz{cfg.nz_WF_import}.npy")
    wig_import = np.load(f"{path_WF}/wig_mar_IST_nz{cfg.nz_WF_import}.npy")
else:
    # raise ValueError('whos_wf must be "davide", "marco", "vincenzo" or "sylvain"')
    raise ValueError('whos_wf must be "marco", at the moment')

bias_selector = cfg.bias_selector

# plot WF to check - they must be IST, not PySSC!
"""
for i in range(10):
    plt.plot(wil_import[:, 0], wil_import[:, i+1], label='wil_import')
plt.title('wil_import')

plt.figure()
for i in range(10):
    plt.plot(wig_import[:, 0], wig_import[:, i+1], label='wig_import')
plt.title('wig_import')
"""

# just a check on the nz of the WF, not very important
assert wig_import[:, 0].size == cfg.nz_WF_import, 'the number of z points in the kernels is not the required one'

k_array = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.k_points)
z_array = np.linspace(z_min, z_max_cl, cfg.zsteps_cl)

# I think the argument names are unimportant, I could have called it k, k_ell
# or k_limber xxx
# this function is to interpolate (2-dimensional) the PS previously generated
# PS = np.genfromtxt(f"{path}/data/Power_Spectrum/Pnl_{units}.txt")  # XXX careful of the units
# PS = np.delete(PS, 0, 1)  # delete first column of PS to adjust the dimensions to (804, 303)
# PS_transposed = PS.transpose()


# Pk
cosmo_classy = csmlb.cosmo_classy

#
Pk = csmlb.calculate_power(cosmo_classy, z_array, k_array, use_h_units=True)

Pk_interp = interp2d(k_array, z_array, Pk)

# ell values

# set the parameters, the functions wants a dict as input
ell_cfg_dict_WL = {
    'nbl': cfg.nbl,
    'ell_min': cfg.ell_min,
    'ell_max': cfg.ell_max_WL,
}

# change ell_max for GC
ell_cfg_dict_GC = ell_cfg_dict_WL.copy()
ell_cfg_dict_GC['ell_max'] = cfg.ell_max_GC

# compute ells using the function in SSC_restructured_v2
ell_LL, _ = ell_utils.ISTF_ells(ell_cfg_dict_WL)
ell_GG, _ = ell_utils.ISTF_ells(ell_cfg_dict_GC)
ell_LG = ell_GG.copy()

cosmo_astropy = csmlb.cosmo_astropy


#
# k_limber_array = csmlb.k_limber(z=1, ell=ell_LL, cosmo_astropy=csmlb.cosmo_astropy, use_h_units=use_h_units)
# P_array = [Pk(csmlb.k_limber(ell, z=1), z=1) for ell in ell_LL]


# IA/noIA, old/new/multibinBias are decided in the import section at the beginning of the code
def wil(z, i):
    # it's i+1: first column is for the redshift array
    wil_interp = interp1d(wil_import[:, 0], wil_import[:, i + 1], kind="linear")
    result_array = wil_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


def wig(z, i):
    # it's i+1: first column is for the redshift array
    wig_interp = interp1d(wig_import[:, 0], wig_import[:, i + 1], kind="linear")
    result_array = wig_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


################## Cijs

###### NEW BIAS ##################
# bias
b = np.zeros((zbins))
for i in range(zbins):
    b[i] = np.sqrt(1 + z_mean[i])


# integrand
def K_ij_LG(z, i, j):
    return wil(z, j) * wig(z, i) / (csmlb.E(z) * csmlb.r(z) ** 2)


def K_ij_GG(z, i, j):
    return wig(z, j) * wig(z, i) / (csmlb.E(z) * csmlb.r(z) ** 2)


# integral
@type_enforced.Enforcer
def Cij_LG_partial(i: int, j: int, zbin: int, ell):
    def integrand(z, i, j, ell):
        return K_ij_LG(z, i, j) * Pk_wrap(kl_wrap(ell, z), z)

    result = c / H0 * quad(integrand, z_minus[zbin], z_plus[zbin], args=(i, j, ell))[0]
    return result


@type_enforced.Enforcer
def Cij_GG_partial(i: int, j: int, zbin: int, ell):
    def integrand(z, i, j, ell):
        return K_ij_GG(z, i, j) * Pk_wrap(kl_wrap(ell, z), z)

    result = c / H0 * quad(integrand, z_minus[zbin], z_plus[zbin], args=(i, j, ell))[0]
    return result


# summing the partial integrals
def sum_Cij_LG(i, j, ell):
    result = 0
    for zbin in range(zbins):
        result += Cij_LG_partial(i, j, zbin, ell) * b[zbin]
    return result


def sum_Cij_GG(i, j, ell):
    result = 0
    for zbin in range(zbins):
        result += Cij_GG_partial(i, j, zbin, ell) * (b[zbin] ** 2)
    return result


###### OLD BIAS ##################
def Pk_wrap(k_ell, z, cosmo_classy=cosmo_classy, use_h_units=use_h_units, Pk_kind='nonlinear', argument_type='scalar'):
    """just a wrapper function to set some args to default values"""
    return csmlb.calculate_power(cosmo_classy, z, k_ell, use_h_units=use_h_units,
                                 Pk_kind=Pk_kind, argument_type=argument_type)


def kl_wrap(ell, z, use_h_units=use_h_units):
    """another simpe wrapper function, so as not to have to rewrite use_h_units=use_h_units"""
    return csmlb.k_limber(ell, z, use_h_units=use_h_units)


def cl_integrand(z, wf_A, wf_B, i, j, ell):
    return ((wf_A(z, i) * wf_B(z, j)) / (csmlb.E(z) * csmlb.r(z) ** 2)) * Pk_wrap(kl_wrap(ell, z), z)


def cl(wf_A, wf_B, i, j, ell):
    """ when used with LG or GG, this implements the "old bias"
    """
    result = c / H0 * quad(cl_integrand, z_min, z_max_cl, args=(wf_A, wf_B, i, j, ell))[0]
    # xxx maybe you can try with scipy.integrate.romberg?
    return result


def reshape(array, npairs, name):
    ind = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/indici.dat")
    ind = ind.astype(int)
    ind = ind - 1  # xxx attenzione!!!

    # setting the ell bounds    
    ind_LL = ind[:55, 2:]  # the ij indices for WL and GC
    ind_LG = ind[55:155, 2:]  # the ij indices for XC

    print(npairs, nbl)  # debug
    output_2D = np.zeros((nbl, npairs))  # creating output 2- dimensional array

    ind_probe = ind_LL  # debug

    if (npairs == 55):
        ind_probe = ind_LL
    elif (npairs == 100):
        ind_probe = ind_LG

    # filling the output array
    for ell in range(nbl):
        for i in range(npairs):
            output_2D[ell, i] = array[ell, ind_probe[i, 0], ind_probe[i, 1]]

    # saving it
    np.savetxt("%s/output/Cij/%s/%s.txt" % (path, cfg.cl_out_folder, name), output_2D)


###############################################################################

# this is now useless. interpolations for the appropriate values will be performed later!
# actually it's better to do it now, to compute 20 values instead of 999
# ell_LL = np.genfromtxt("%s/output/ell_values/ell_WL.dat" %path)


# ell_LG = np.genfromtxt("%s/output/ell_values/ell_XC.dat" %path)
# ell_GG = np.genfromtxt("%s/output/ell_values/ell_GC.dat" %path)

# these are Vincenzo's ell values: 
# ell_min = 10
# ell_max = 5000
# ell_steps = 999
# ell_values = np.linspace(ell_min, ell_max, ell_steps)

def compute_cl(Cij_function, wf_A, wf_B, ell_values, symmetric_flag):
    Cij_array = np.zeros((nbl, zbins, zbins))
    k = 0
    for ell in ell_values:
        if ell > ell_values[0]:  # the first time k should remain 0
            k = k + 1
        print("k = %i, ell = %f" % (k, ell))
        print("the program took %i seconds to run" % (time.perf_counter() - script_start))

        for i in range(zbins):
            for j in range(zbins):
                if symmetric_flag == "yes":
                    if j >= i:  # C_LL and C_GG are symmetric!
                        Cij_array[k, i, j] = Cij_function(wf_A, wf_B, i, j, ell)
                else:
                    Cij_array[k, i, j] = Cij_function(wf_A, wf_B, i, j, ell)

    return Cij_array


def compute_cl_v2(Cij_function, wf_A, wf_B, ell_values, symmetric_flag):
    Cij_array = np.zeros((nbl, zbins, zbins))

    if symmetric_flag == "yes":
        for ell_idx, ell_val in enumerate(ell_values):
            for i in range(zbins):
                for j in range(i, zbins):
                    Cij_array[ell_idx, i, j] = Cij_function(wf_A, wf_B, i, j, ell_val)

    else:
        for ell_idx, ell_val in enumerate(ell_values):
            for i in range(zbins):
                for j in range(zbins):  # this line is different
                    Cij_array[ell_idx, i, j] = Cij_function(wf_A, wf_B, i, j, ell_val)

    return Cij_array


@njit
def fill_symmetric_Cls(Cij_array):
    for k in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                if j < i:  # C_LL and C_GG are symmetric!
                    Cij_array[k, i, j] = Cij_array[k, j, i]
    return Cij_array


def check_k_limber():
    for ell in ell_GG:
        for z in np.linspace(0.03, 4, 100):
            if kl_wrap(ell, z) > 30:
                print(ell, z, kl_wrap(ell, z))

###############################################################################
################# end of function declaration
###############################################################################

# XXX I just computed LL, to be quicker
# compute
C_LL_array = compute_cl(cl, wil, wil, ell_LL, symmetric_flag="yes")
# if bias_selector == "newBias":
#     C_GG_array = compute_cl(ell_GG, sum_Cij_GG, symmetric_flag="yes")
#     C_LG_array = compute_cl(ell_LG, sum_Cij_LG, symmetric_flag="no")
# elif bias_selector == "oldBias":
#     C_GG_array = compute_cl(ell_GG, sum_Cij_GG, symmetric_flag="yes")
#     C_LG_array = compute_cl(ell_LG, sum_Cij_LG, symmetric_flag="no")
# else:
#     raise ValueError('bias_selector must be newBias or oldBias')

# symmetrize
# C_LL_array = fill_symmetric_Cls(C_LL_array)
# C_GG_array = fill_symmetric_Cls(C_GG_array)

# import Vincenzo to check:
path_vinc = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Cij-NonLin-eNLA_15gen'
C_LL_vinc = np.genfromtxt(f'{path_vinc}/CijLL-LCDM-NonLin-eNLA.dat')
C_LG_vinc = np.genfromtxt(f'{path_vinc}/CijLG-LCDM-NonLin-eNLA.dat')
C_GG_vinc = np.genfromtxt(f'{path_vinc}/CijGG-LCDM-NonLin-eNLA.dat')

dav = C_LL_array[:, 0, 0]
vinc = C_LL_vinc
ell_dav = ell_LL
ell_vinc = vinc[:, 0]

# plot my array, vincenzo's cls and the % difference
cl_func = interp1d(ell_vinc, vinc[:, 1])
cl_interp = cl_func(ell_dav)
diff = mm.percent_diff(dav, cl_interp)

plt.plot(ell_dav, dav, label='dav')
plt.plot(vinc[:, 0], vinc[:, 1], label='cl_vinc')
plt.plot(ell_dav, diff, label='diff')
plt.legend()
plt.xscale('log')
plt.yscale('log')

# test the Pk array
# z_array_limber = np.linspace(0.001, 4, 100)
#
# for zi, zval in enumerate(z_array_limber):
#     for ell_idx, ell_val in enumerate(ell_LL):
#
#         # k_limber should already be in the correct units, from the cosmo_astropy call
#         kl = csmlb.k_limber(zval, ell_LL, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
#
#         # Pk_with_classy_clustertlkt wants in input k in 1/Mpc; so, if I'm using h units, transform kl to 1/Mpc
#         # ! this is assuming that the k_limber function returns kl in the correct units
#         if use_h_units:
#             kl *= h
#
#         kl[], P_kl_z = csmlb.Pk_with_classy_clustertlkt(cosmo, zval, kl, use_h_units, Pk_kind='nonlinear',
#                                                         argument_type='scalar')


# save
np.save(project_path / f"output/Cij/{cfg.cl_out_folder}/Cij_LL.npy", C_LL_array)
np.save(project_path / f"output/Cij/{cfg.cl_out_folder}/Cij_LG.npy", C_LG_array)
np.save(project_path / f"output/Cij/{cfg.cl_out_folder}/Cij_GG.npy", C_GG_array)

print("saved")
############### reshape to compare with others ##########
# reshape(C_LL_array, 55, "C_LL_2D.txt")
# reshape(C_LG_array, 100, "C_LG_2D.txt")
# reshape(C_GG_array, 55, "C_GG_2D.txt")

print("the program took %i seconds to run" % (time.perf_counter() - script_start))
