import sys
import copy
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.special import erf
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent

my_module_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/lib'
sys.path.append(my_module_path)
import my_module as mm

matplotlib.use('Qt5Agg')

plot_params = {'lines.linewidth': 3.5,
               'font.size': 20,
               'axes.labelsize': 'x-large',
               'axes.titlesize': 'x-large',
               'xtick.labelsize': 'x-large',
               'ytick.labelsize': 'x-large',
               'mathtext.fontset': 'stix',
               'font.family': 'STIXGeneral',
               'figure.figsize': (8, 8)
               # 'backend': 'Qt5Agg'
               }
plt.rcParams.update(plot_params)
markersize = 10

start_time = time.perf_counter()


########################################################################################################################
########################################################################################################################
########################################################################################################################

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
                m, c = np.polyfit(variations_arr_cpy, Cl_arr_cpy[:, i, j, ell], deg=1)
                fitted_y_values = m * variations_arr_cpy + c

                # check % difference
                perc_diffs = mm.percent_diff(Cl_arr_cpy[:, i, j, ell], fitted_y_values)

                # as long as any element has a percent deviation greater than 1%, remove first and last values
                while np.any(perc_diffs > 1):
                    print('removing first and last values, perc_diffs array:', perc_diffs)

                    # the condition is satisfied, removing the first and last values
                    Cl_arr_cpy = np.delete(Cl_arr_cpy, [0, -1], axis=0)
                    variations_arr_cpy = np.delete(variations_arr_cpy, [0, -1])

                    # re-compute the fit on the reduced set
                    m, c = np.polyfit(variations_arr_cpy, Cl_arr_cpy[:, i, j, ell], deg=1)
                    fitted_y_values = m * variations_arr_cpy + c

                    # test again
                    perc_diffs = mm.percent_diff(Cl_arr_cpy[:, i, j, ell], fitted_y_values)

                    # plt.figure()
                    # plt.plot(Omega_c_values_toder, fitted_y_values, '--', lw=2, c=colors[iteration])
                    # plt.plot(Omega_c_values_toder, CLL_toder[:, i, j, ell], marker='o', c=colors[iteration])

                # store the value of the derivative
                dCLL_arr[i, j, ell] = m

    return dCLL_arr


########################################################################################################################
########################################################################################################################
########################################################################################################################

# TODO understand whether passing pk = None to the CLL call is enough to account for the different cosmological parameters

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
params_names_LL = ["Om", "Ob", "wz", "wa", "h", "ns", "s8", "Aia", "eIA", "bIA"]
params_names_XC = params_names_LL + ["bL01", "bL02", "bL03", "bL04", "bL05", "bL06", "bL07", "bL08",
                                     "bL09", "bL10"]

# this is immutable
fiducial_params = {'Om': 0.32,
                   'Ob': 0.05,
                   'wz': -1.0,
                   'wa': 0.0,
                   'h': 0.67,
                   'ns': 0.96,
                   's8': 0.815583,
                   'Aia': 1.72,
                   'eIA': -0.41,
                   'bIA': 2.17,
                   # ... add bias parameters and IA parameters
                   }

# this is varied
free_params = fiducial_params.copy()

fixed_params = {'m_nu': 0.06}

percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
                          0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100
num_variations = len(percentages)

# dictionary storing the perturbed values of the parameters around the fiducials
variations = {}
for key in free_params.keys():
    variations[key] = free_params[key] * (1 + percentages)

# wa = 0, so the deviations are the percentages themselves
variations['wa'] = percentages

# choose the ell binning
which_ells = 'IST-F'
ell_min = 10
ell_max_WL = 5000
nbl = 30

if which_ells == 'IST-F':
    # IST:F recipe:
    ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
    l_centr_WL = (ell_WL[1:] + ell_WL[:-1]) / 2
    logarithm_WL = np.log10(l_centr_WL)
    ell_WL = logarithm_WL
    l_lin_WL = 10 ** ell_WL
    ell = l_lin_WL
elif which_ells == 'IST-NL':
    # this is slightly different
    ell_bins = np.linspace(np.log(ell_min), np.log(ell_max_WL), nbl + 1)
    ell = (ell_bins[:-1] + ell_bins[1:]) / 2.
    ell = np.exp(ell)
else:
    raise ValueError('Wrong choice of ell bins: which_ells must be either IST-F or IST-NL, and '
                     'nbl must be 20.')

# redshift array ("ztab")
zmin, zmax, dz = 0.001, 2.5, 0.001
zmax_marco, znum_marco = 4, 10_000

z_stef = np.arange(zmin, zmax, dz)
z_marco = np.linspace(zmin, zmax_marco, znum_marco)

ztab = z_stef

zi = np.array([[zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
               [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]])

zbins = len(zi[0])

z_minus = zi[0, :]  # lower edge of z bins
z_plus = zi[1, :]  # upper edge of z bins
z_mean = (z_minus + z_plus) / 2  # cener of the z bins

# get number of redshift pairs
npairs_auto, npairs_asimm, npairs_3x2pt = mm.get_zpairs(zbins)


# z distribution
nzEuclid = 30 * (ztab / 0.9 * np.sqrt(2)) ** 2 * np.exp(-(ztab / 0.9 * np.sqrt(2)) ** 1.5)

fout, cb, zb, sigmab, c0, z0, sigma0 = 0.1, 1, 0, 0.05, 1, 0.1, 0.05

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb *
                      (cb * fout *
                       (erf((ztab - z0 - c0 * zi[0, iz]) / np.sqrt(2) / (1 + ztab) / sigma0) -
                        erf((ztab - z0 - c0 * zi[1, iz]) / np.sqrt(2) / (1 + ztab) / sigma0)) + c0 * (1 - fout) *
                       (erf((ztab - zb - cb * zi[0, iz]) / np.sqrt(2) / (1 + ztab) / sigmab) -
                        erf((ztab - zb - cb * zi[1, iz]) / np.sqrt(2) / (1 + ztab) / sigmab))) for iz in range(zbins)])

# normalize n_i(z)
for i in np.arange(zbins):
    norm_factor = np.sum(nziEuclid[i, :]) * dz
    nziEuclid[i, :] /= norm_factor

# IA parameters
IAFILE = np.genfromtxt(project_path / 'input/scaledmeanlum-E2Sa.dat')
CIA = 0.0134

# galaxy bias: construct bias array
b_array = np.asarray([bias(z, zi) for z in ztab])



# declare cl and dcl vectors
CLL = {}
dCLL = {}

# loop over the free parameters and store the cls in a dictionary
for free_param_name in free_params.keys():

    # instantiate derivatives array for the given free parameter
    CLL[free_param_name] = np.zeros((num_variations, zbins, zbins, nbl))

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
                              # to cross w = -1
                              extra_parameters={"camb": {"dark_energy_model": "DarkEnergyPPF"}}
                              )

        # Computes the WL (w/ and w/o IAs) and GCph kernels
        A_IA, eta_IA, beta_IA = free_params['Aia'], free_params['eIA'], free_params['bIA']
        FIAzNoCosmoNoGrowth = - A_IA * CIA * (1 + IAFILE[:, 0]) ** eta_IA * IAFILE[:, 1] ** beta_IA

        FIAz = FIAzNoCosmoNoGrowth * \
               (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / \
               ccl.growth_factor(cosmo, 1 / (1 + IAFILE[:, 0]))

        wil = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
               for iz in range(zbins)]

        wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
                                              mag_bias=None) for iz in range(zbins)]

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
        # Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

        # the key specifies the parameter, but I still need an array of values - corresponding to the 15 variations over
        # the fiducial values
        CLL[free_param_name][variation_idx, :, :, :] = np.array([[ccl.angular_cl(cosmo, wil[iz], wil[jz],
                                                                                 ell, p_of_k_a=None)
                                                                  for iz in range(zbins)]
                                                                 for jz in range(zbins)])

        print(
            f'{free_param_name} = {free_params[free_param_name]:.4f} Cls computed in {(time.perf_counter() - t0):.2f} '
            f'seconds')

    # once finished looping over the variations, reset the parameter to its fiducial value
    free_params[free_param_name] = fiducial_params[free_param_name]

    # save the Cls
    dCLL[free_param_name] = stem(CLL[free_param_name], variations[free_param_name], zbins, nbl)
    print(f'SteM derivative computed for {free_param_name}')

    # np.save(project_path / f'output/dCLLd{free_param_name}.npy', dCLL[free_param_name])



assert 1 > 2
########################################################################################################################
# ! new code

# get the values from the object
z_bin = 4

b_array = np.asarray([bias(z, zi) for z in ztab])
wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
                                      mag_bias=None) for iz in range(zbins)]

a_arr = 1 / (1 + ztab[::-1])
chi = ccl.background.comoving_radial_distance(cosmo, a_arr[::-1])  # in Mpc
wil_values = wil[z_bin].get_kernel(chi=chi)
wig_values = wig[z_bin].get_kernel(chi=chi)[0] * b_array

# TODO set it manually? this doesn't work...
wig_bias_corrected = ccl.tracers.Tracer.add_tracer(cosmo, kernel=(chi, wig_values),
                                                   transfer_ka=None, transfer_k=None, transfer_a=None,
                                                   der_bessel=0, der_angles=0,
                                                   is_logt=False, extrap_order_lok=0, extrap_order_hik=2)

plt.plot(ztab, wig_values, '.-', lw=2, label='wig_values')

# ! validate kernels
cosmocentral_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/input/CosmoCentral_outputs/WF'
wil_marco = np.load(f'{cosmocentral_path}/WL_WeightFunction_bia2.17.npy').T
wig_marco = np.load(f'{cosmocentral_path}/GC_WeightFunction.npy').T
vincenzo_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/config/common_data/everyones_WF_from_Gdrive/vincenzo'
wig_vincenzo = np.genfromtxt(f'{vincenzo_path}/wig_vincenzo_IST_nz300.dat')
davide_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/config/common_data/everyones_WF_from_Gdrive/davide/nz10000/gen2022'
wil_davide = np.genfromtxt(f'{davide_path}/wil_dav_IA_IST_nz10000_bia2.17.txt')
wig_davide = np.genfromtxt(f'{davide_path}/wig_dav_IST_nz10000.txt')

wig_vincenzo = wig_vincenzo[1:, :]  # remove 1st row, z = 0
z_vinc = wig_vincenzo[:, 0]
wig_dav_fn = interp1d(wig_davide[:, 0], wig_davide[:, z_bin + 1], kind='linear')
wig_dav = wig_dav_fn(z_vinc)

z_marco = np.linspace(0.001, 4, 10_000)
diff = mm.percent_diff(wig_davide[:, z_bin + 1], wig_marco[:, z_bin])
plt.plot(z_marco, diff)

# plt.plot(ztab, np.abs(wig_values[0]), label='pyCCL')
plt.plot(z_marco, np.abs(wig_marco[:, z_bin]), label='marco')
# plt.plot(wig_vincenzo[:, 0], np.abs(wig_vincenzo[:, z_bin + 1]), label='vincenzo')
plt.plot(wig_davide[:, 0], np.abs(wig_davide[:, z_bin + 1]), '--', label='davide')
plt.legend()
plt.grid()
plt.xlabel('z')
# plt.yscale('log')
