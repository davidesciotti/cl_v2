import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import time
from pathlib import Path
from numba import jit
import sys

# get project directory
path = Path.cwd().parent.parent

sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

path = "/Users/davide/Documents/Lavoro/Programmi/Cij_davide"
path_SSC = "/Users/davide/Documents/Lavoro/Programmi/SSC_comparison"

start = time.time()

c = 299792.458  # km/s
H0 = 67  # km/(s*Mpc)
Om0 = 0.32
Ode0 = 0.68
Ox0 = 0

z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))

z_mean = (z_plus + z_minus) / 2
z_min = z_minus[0]
z_max = z_plus[9]
# xxx is z_max = 4 to be used everywhere?
# z_max   = 4 # XXX this has been removed 

zbins = 10
nbl = 30

h = 0.67

zsteps = 303
# z_array = np.linspace(z_min, z_max,   zsteps)
z_array = np.linspace(0, 2.5, zsteps)  # not ok

k_min = -5.442877
k_max = 34.03235477928787  # in Mpc**-1
k_points = 804
k_array = np.logspace(k_min, np.log10(k_max), k_points)
# TODO fix z_array -> fatto?


bias_selector = "newBias"
IA_flag = "IA"
units = "1Mpc"
z_max = 2.5  # XXXX see p.25 of IST paper
nz = 10000
Cij_output_folder = f"Cijs_v19_ALL/Cij_WFdavide_{IA_flag}_{bias_selector}_nz{nz}_{units}"  # this needs to be fixed
wil_import = np.genfromtxt(
    f"{path_SSC}/data/everyones_WFs/everyones_WF_from_Gdrive/davide/nz{nz}/wil_{IA_flag}_IST_nz{nz}.txt")
wig_import = np.genfromtxt(
    f"{path_SSC}/data/everyones_WFs/everyones_WF_from_Gdrive/davide/nz{nz}/wig_IST_nz{nz}.txt")


################### defining the functions
def E(z):
    result = np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
    return result


def inv_E(z):
    result = 1 / np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
    return result


def r(z):
    # r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    result = c / H0 * quad(inv_E, 0, z)[0]  # integrate 1/E(z) from 0 to z
    return result


def k_limber(ell, z):
    return (ell + 1 / 2) / r(z)


# I think the argument names are unimportant, I could have called it k, k_ell
# or k_limber xxx
# this function is to interpolate (2-dimensional) the PS previously generated
# PS = np.genfromtxt(f"{path}/data/Power_Spectrum/Pnl_{units}.txt")  # XXX careful of the units
# PS = np.delete(PS, 0, 1)  # delete first column of PS to adjust the dimensions to (804, 303)
# PS_transposed = PS.transpose()

PS = np.load(f"/jobs/SSC_comparison/output/Pk/Pk_kind=nonlinear_hunits=True.npy")  # XXX careful of the units
z_pk = np.load(f"/jobs/SSC_comparison/output/Pk/z_array.npy")  # XXX careful of the units
k_pk = np.load(f"/jobs/SSC_comparison/output/Pk/k_array_hunits=True.npy")  # XXX careful of the units



def P(k_ell, z):
    # order: interp2d(k, z, PS, kind = "linear") first x, then y
    # is is necessary to transpose, the PS array must have dimensions
    # [z.ndim, k.ndim]. I have it the other way round. The console says:
    # ValueError: When on a regular grid with x.size = m and y.size = n,
    # if z.ndim == 2, then z must have shape (n, m)
    PS_interp = interp2d(k_pk, z_pk, PS, kind="linear")
    result_array = PS_interp(k_ell, z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array OK!
    return result


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
PS = np.genfromtxt(f"{path}/data/Power_Spectrum/Pnl_{units}.txt")  # XXX careful of the units

z_0 = (PS_vinc[:, 1] == 0)  # this is where redshift = 0
PS_vinc_z0 = PS_vinc[z_0]

plt.plot(np.log10(PS[:, 0]), PS[:, 1])
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
    return wil(z, j) * wig(z, i) / (E(z) * r(z) ** 2)


def K_ij_GG(z, i, j):
    return wig(z, j) * wig(z, i) / (E(z) * r(z) ** 2)


# integral
def Cij_LG_partial(i, j, nbin, ell):
    def integrand(z, i, j, ell):
        return K_ij_XC(z, i, j) * P(k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_minus[nbin], z_plus[nbin], args=(i, j, ell))[0]
    return result


def Cij_GG_partial(i, j, nbin, ell):
    def integrand(z, i, j, ell):
        return K_ij_GG(z, i, j) * P(k_limber(ell, z), z)

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
        return ((wil(z, i) * wil(z, j)) / (E(z) * r(z) ** 2)) * P(k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_min, 4, args=(i, j, ell))[0]
    # check z_min, z_max xxx
    return result


def Cij_LG_function(i, j, ell):  # xxx GL or LG? OLD BIAS
    def integrand(z, i, j, ell):  # first argument is the integration variable
        return ((wil(z, i) * wig(z, j)) / (E(z) * r(z) ** 2)) * P(k_limber(ell, z), z)

    result = c / H0 * quad(integrand, z_min, 4, args=(i, j, ell))[0]
    return result


def Cij_GG_function(i, j, ell):  # OLD BIAS
    def integrand(z, i, j, ell):  # first argument is the integration variable
        return ((wig(z, i) * wig(z, j)) / (E(z) * r(z) ** 2)) * P(k_limber(ell, z), z)

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
        print("the program took %i seconds to run" % (time.time() - start))
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

winsound.Beep(frequency=440, duration=2000)
print("the program took %i seconds to run" % (time.time() - start))
