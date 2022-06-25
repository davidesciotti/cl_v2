import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time
from pathlib import Path
from numba import jit
import sys

# get project directory
project_path = Path.cwd().parent
path_SSC = Path.cwd().parent.parent / "SSC_restructured_v2"

sys.path.append(str(project_path))
sys.path.append(str(Path.cwd().parent.parent))

import SSC_restructured_v2.lib.my_module as mm
import config.config as cfg
import lib.cosmo_lib as csmlb

script_start = time.perf_counter()

H0 = cfg.H0
c = cfg.c
Om0 = cfg.Om0
Ode0 = cfg.Ode0
Ox0 = cfg.Ox0
w0 = cfg.w0
wa = cfg.wa
Neff = cfg.Neff
m_nu = cfg.m_nu
Ob0 = cfg.Ob0

z_minus = cfg.z_minus
z_plus = cfg.z_plus

z_mean = cfg.z_mean
z_min = cfg.z_min
z_max = cfg.z_max

zbins = cfg.zbins
nbl = cfg.nbl

h = cfg.h

zsteps = 303
# z_array = np.linspace(z_min, z_max,   zsteps)
z_array = np.linspace(0, 2.5, zsteps)  # not ok

# TODO check k and z arrays
k_min = 10 ** (-5.442877)
k_max = 30  # in Mpc**-1
k_points = 804
k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_points)
# TODO fix z_array -> fatto?
z_array = np.linspace(z_min, z_max, cfg.znum_cl)


if cfg.useIA:
    IA_flag = "IA"
elif not cfg.useIA:
    IA_flag = 'noIA'

units = cfg.units

if units == '1/Mpc':
    use_h_units = False
elif units == 'h/Mpc':
    use_h_units = True
else:
    raise ValueError('units must be 1/Mpc or h/Mpc')

# z_max = 2.5  # XXXX see p.25 of IST paper
nz = 8000

Cij_output_folder = f"cl_v20/Cij_WFdavide_{IA_flag}_nz{nz}_{units}"  # this needs to be fixed

if cfg.whos_wf == 'vincenzo':
    wil_import = np.genfromtxt(
        f"{path_SSC}/config/common_data/everyones_WF_from_Gdrive/vincenzo/wil_vincenzo_IA_PySSC_nz8000.dat")
    wig_import = np.genfromtxt(
        f"{path_SSC}/config/common_data/everyones_WF_from_Gdrive/vincenzo/wig_vincenzo_PySSC_nz8000.dat")
else:
    raise ValueError('whos_wf must be \'vincenzo\'')

z_wf = wig_import[:, 0]

################### defining the functions

# I think the argument names are unimportant, I could have called it k, k_ell
# or k_limber xxx
# this function is to interpolate (2-dimensional) the PS previously generated
# PS = np.genfromtxt(f"{path}/data/Power_Spectrum/Pnl_{units}.txt")  # XXX careful of the units
# PS = np.delete(PS, 0, 1)  # delete first column of PS to adjust the dimensions to (804, 303)
# PS_transposed = PS.transpose()

cosmo_classy = csmlb.cosmo_classy

Pk = csmlb.calculate_power(cosmo_classy, z_array, k_array, use_h_units=True, Pk_kind='nonlinear',
                           argument_type='arrays')
Pk_interp = interp2d(k_array, z_array, Pk)



assert 1 > 2





############################################### DEBUGGIN'

path_SSC_CMBX = "/jobs/SSC_CMBX"
ell_LL = np.genfromtxt(f"{path_SSC_CMBX}/misc/notebooks/inputs/ell_values/ell_WL_ellMaxWL5000_nbl{nbl}.txt")
ell_GG = np.genfromtxt(f"{path_SSC_CMBX}/misc/notebooks/inputs/ell_values/ell_GC_ellMaxGC3000_nbl{nbl}.txt")

# XXX fundamental
ell_LL = 10 ** ell_LL
ell_GG = 10 ** ell_GG
ell_LG = ell_GG

P_array = [P(k_limber(ell, z=1), z=1) for ell in ell_LL]
k_limber_array = [k_limber(ell, z=1) for ell in ell_LL]

z_dav = np.genfromtxt("%s/data/Power_Spectrum/z.txt" % path)

# PS vincenzo:
PS_vinc = np.genfromtxt(f"{path_SSC_CMBX}/data/vincenzo/Cij-NonLin-eNLA_15gen/Pnl-TB-LCDM.dat")
z_vinc = np.unique(PS_vinc[:, 1])

# PS davide: reimport it to have the first column
Pk = np.genfromtxt(f"{path}/data/Power_Spectrum/Pnl_{units}.txt")  # XXX careful of the units

z_0 = (PS_vinc[:, 1] == 0)  # this is where redshift = 0
PS_vinc_z0 = PS_vinc[z_0]

plt.plot(np.log10(Pk[:, 0]), Pk[:, 1])
# 10** because I already take the log scale on the y axis
plt.plot(PS_vinc_z0[:, 0], 10 ** PS_vinc_z0[:, 2], '--')  # column 2 should be the nonlin PS
# plt.plot(ell_LL, k_limber_array)
plt.plot(np.log10(ell_LL), P_array)

plt.yscale("log")

sys.exit()


############################################### STOP DEBUGGIN'


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
def K_ij_XC(z, i, j):
    return wil(z, j) * wig(z, i) / (csmlb.E(z) * csmlb.r(z) ** 2)


def K_ij_GG(z, i, j):
    return wig(z, j) * wig(z, i) / (csmlb.E(z) * csmlb.r(z) ** 2)


# integral
def Cij_LG_partial(i, j, nbin, ell):
    def integrand(z, i, j, ell):
        return K_ij_XC(z, i, j) * P(csmlb.k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_minus[nbin], z_plus[nbin], args=(i, j, ell))[0]
    return result


def Cij_GG_partial(i, j, nbin, ell):
    def integrand(z, i, j, ell):
        return K_ij_GG(z, i, j) * P(csmlb.k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_minus[nbin], z_plus[nbin], args=(i, j, ell))[0]
    return result


# summing the partial integrals
def sum_Cij_LG(i, j, ell):
    result = 0
    for nbin in range(zbins):
        result += Cij_LG_partial(i, j, nbin, ell) * b[nbin]
    return result


def sum_Cij_GG(i, j, ell):
    result = 0
    for nbin in range(zbins):
        result += Cij_GG_partial(i, j, nbin, ell) * (b[nbin] ** 2)
    return result


###### OLD BIAS ##################
def Cij_LL_function(i, j, ell):
    def integrand(z, i, j, ell):  # first argument is the integration variable
        return ((wil(z, i) * wil(z, j)) / (csmlb.E(z) * csmlb.r(z) ** 2)) * P(csmlb.k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_min, 4, args=(i, j, ell))[0]
    # check z_min, z_max xxx
    return result


def Cij_LG_function(i, j, ell):  # xxx GL or LG? OLD BIAS
    def integrand(z, i, j, ell):  # first argument is the integration variable
        return ((wil(z, i) * wig(z, j)) / (csmlb.E(z) * csmlb.r(z) ** 2)) * P(csmlb.k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_min, 4, args=(i, j, ell))[0]
    return result


def Cij_GG_function(i, j, ell):  # OLD BIAS
    def integrand(z, i, j, ell):  # first argument is the integration variable
        return ((wig(z, i) * wig(z, j)) / (csmlb.E(z) * csmlb.r(z) ** 2)) * P(csmlb.k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_min, 4, args=(i, j, ell), full_output=True)[0]
    # xxx maybe you can try with scipy.integrate.romberg
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
    np.savetxt("%s/output/Cij/%s/%s.txt" % (path, Cij_output_folder, name), output_2D)


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

def compute_Cij(ell_values, Cij_function, symmetric_flag):
    Cij_array = np.zeros((nbl, zbins, zbins))
    k = 0
    for ell in ell_values:
        if ell > ell_values[0]:  # the first time k should remain 0
            k = k + 1
        print("k = %i, ell = %f" % (k, ell))
        print("the program took %i seconds to run" % (time.time() - script_start))
        for i in range(zbins):
            for j in range(zbins):
                if symmetric_flag == "yes":
                    if j >= i:  # C_LL and C_GG are symmetric!
                        Cij_array[k, i, j] = Cij_function(i, j, ell)
                else:
                    Cij_array[k, i, j] = Cij_function(i, j, ell)

    return Cij_array


@jit(nopython=True)
def fill_symmetric_Cls(Cij_array):
    for k in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                if j < i:  # C_LL and C_GG are symmetric!
                    Cij_array[k, i, j] = Cij_array[k, j, i]
    return Cij_array


###############################################################################
################# end of function declaration
###############################################################################
path_masterUnified_forSSCcomparison = "C:/Users/dscio/Documents/Lavoro/Programmi/master_unified_forSSCcomparison"
ell_LL = np.genfromtxt(f"{path_masterUnified_forSSCcomparison}/output/ell_values/ell_WL_ellMaxWL5000_nbl{nbl}.txt")
ell_GG = np.genfromtxt(f"{path_masterUnified_forSSCcomparison}/output/ell_values/ell_GC_ellMaxGC3000_nbl{nbl}.txt")

# XXX fundamental 
ell_LL = 10 ** ell_LL
ell_GG = 10 ** ell_GG
ell_LG = ell_GG

# XXX I just computed LL, to be quicker
# compute
C_LL_array = compute_Cij(ell_LL, Cij_LL_function, symmetric_flag="yes")
# if bias_selector == "newBias":
#     C_GG_array = compute_Cij(ell_GG, sum_Cij_GG, symmetric_flag = "yes") 
#     C_LG_array = compute_Cij(ell_LG, sum_Cij_LG, symmetric_flag = "no")
# elif bias_selector == "oldBias":
#     C_GG_array = compute_Cij(ell_GG, sum_Cij_GG, symmetric_flag = "yes") 
#     C_LG_array = compute_Cij(ell_LG, sum_Cij_LG, symmetric_flag = "no")

# symmetrize
# C_LL_array = fill_symmetric_Cls(C_LL_array)
# C_GG_array = fill_symmetric_Cls(C_GG_array)

# save
# np.save("%s/output/Cij/%s/Cij_LL.npy" %(path, Cij_output_folder), C_LL_array)
# np.save("%s/output/Cij/%s/Cij_LG.npy" %(path, Cij_output_folder), C_LG_array)
# np.save("%s/output/Cij/%s/Cij_GG.npy" %(path, Cij_output_folder), C_GG_array)
print("saved")
############### reshape to compare with others ##########
# reshape(C_LL_array, 55, "C_LL_2D.txt")
# reshape(C_LG_array, 100, "C_LG_2D.txt")
# reshape(C_GG_array, 55, "C_GG_2D.txt")

print("the program took %i seconds to run" % (time.time() - script_start))
