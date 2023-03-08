import gc
import sys
import time
import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import pyccl as ccl
from scipy.interpolate import interp1d

project_path = Path.cwd().parent

sys.path.append(f'{project_path}/config')
import config_wlcl as cfg
import config_ISTF as cfg_ISTF

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

sys.path.append(f'{project_path.parent}/SSC_restructured_v2/bin')
import ell_values
import covariance as cov_lib

import wf_cl_lib

script_start = time.perf_counter()

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! config stuff
general_cfg = cfg_ISTF.general_cfg
covariance_cfg = cfg_ISTF.covariance_cfg

zbins = cfg.zbins
z_grid = cfg.z_grid
zpoints = len(z_grid)
WFs_output_folder = 'feb_2023'
# ! end config

# my wf
wig_IST = wf_cl_lib.wig_IST(z_grid, 'with_galaxy_bias')
gal_bias_2d_array = wf_cl_lib.wig_IST(z_grid, 'without_galaxy_bias')
wil_IA_IST = wf_cl_lib.wil_final(z_grid, 'with_IA')
# wil_noIA_IST = wf_cl_lib.wil_final(z_grid, 'without_IA')
# wil_IAonly_IST = wf_cl_lib.wil_final(z_grid, 'IA_only')

# interpolators for the cls
wil_IA_IST_func = interp1d(z_grid, wil_IA_IST, axis=0, kind='linear')
wig_IST_func = interp1d(z_grid, wig_IST, axis=0, kind='linear')

# wf from PyCCL
wil_PyCCL = wf_cl_lib.wil_PyCCL(z_grid, 'with_IA')
wig_PyCCL = wf_cl_lib.wig_PyCCL(z_grid, 'with_galaxy_bias')

# set rainbow colormap over zbins values
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

# check wil
# diff = mm.percent_diff(wil_IA_IST, wil_PyCCL)
# plt.figure()
# for i in range(zbins):
#     plt.plot(z_grid, wil_PyCCL[:, i], label=f"wil tot i={i}", c=colors[i], ls='-')
#     plt.plot(z_grid, wil_IA_IST[:, i], label=f"wil tot i={i}", c=colors[i], ls='-')
#     plt.plot(z_grid, diff[:, i], label=f"perc diff i={i}", c=colors[i], ls='-')
# plt.legend()
# plt.grid()
# plt.show()

# check wig
# diff = mm.percent_diff(wig_PyCCL, wig_IST)
# plt.figure()
# for i in range(zbins):
#     plt.plot(z_grid, wig_IST[:, i], label=f"wig i={i}", c=colors[i], ls='-')
#     plt.plot(z_grid, wig_PyCCL[:, i], label=f"wig i={i}", c=colors[i], ls='--')
#     plt.plot(z_grid, diff[:, i], label=f"perc diff i={i}", c=colors[i], ls='-')
# plt.legend()
# plt.grid()
# plt.show()

# as well as their "sub-components":
# save everything:
wf_output_path = f'{project_path}/output/WF/{WFs_output_folder}'
wf_benchmarks_path = wf_output_path + '/benchmarks'
np.save(f'{wf_output_path}/z_grid.npy', z_grid)
np.save(f'{wf_output_path}/wil_IA_IST_nz{zpoints}.npy', wil_IA_IST)
# np.save(f'{wf_output_path}/wil_noIA_IST_nz{zpoints}.npy', wil_noIA_IST)
# np.save(f'{wf_output_path}/wil_IAonly_IST_nz{zpoints}.npy', wil_IAonly_IST)
np.save(f'{wf_output_path}/wig_IST_nz{zpoints}.npy', wig_IST)
np.save(f'{wf_output_path}/wig_nobias_IST_nz{zpoints}.npy', wig_IST / gal_bias_2d_array)

# ! VALIDATION against FS1
# wig_fs1 = np.genfromtxt(
#     '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1/KernelFun/WiGC-EP10.dat')
# wil_fs1 = np.genfromtxt(
#     '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1/KernelFun/WiWL-EP10.dat')
# zvalues_fs1 = wig_fs1[:, 0]

# for zbin_idx in range(zbins):
#     plt.plot(zvalues_pyccl, wig_pyccl[:, zbin_idx], label='wig pyccl')
#     plt.plot(z_array, wig_IST_arr[:, zbin_idx + 1], label='wig davide', ls='--')
# plt.legend()
# plt.grid()

# plt.figure()
# for zbin_idx in range(zbins):
#     plt.plot(zvalues_fs1, wig_fs1[:, zbin_idx + 1], label='wig fs1')
#     plt.plot(z_arr, wig_IST_arr[:, zbin_idx + 1], label='wig davide', ls='--')
# plt.legend()
# plt.grid()
#
# plt.figure()
# for zbin_idx in range(zbins):
#     plt.plot(zvalues_fs1, wil_fs1[:, zbin_idx + 1], label='wil fs1')
#     plt.plot(z_arr, wil_IA_IST_arr[:, zbin_idx + 1], label='wil davide', ls='--')
# plt.legend()
# plt.grid()


# Import fiducial P(k,z)
PkFILE = np.genfromtxt(project_path / 'data/pkz-Fiducial.txt')

# ! XXX are the units correct?
# Populate vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]

cosmo = wf_cl_lib.instantiate_ISTFfid_PyCCL_cosmo_obj()
zlist = np.unique(PkFILE[:, 0])
k_points = int(len(PkFILE[:, 2]) / len(zlist))
klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
z_points = len(zlist)
Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)  # it's the natural log, not log10
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

z_values = ISTFfid.photoz_bins['z_mean']
bias_values = np.asarray([wf_cl_lib.b_of_z(z) for z in z_values])
gal_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values, z_values, zbins, z_grid, 'constant')

wil_PyCCL_obj = wf_cl_lib.wil_PyCCL(z_grid, 'with_IA', cosmo=None, return_PyCCL_object=True)
wig_PyCCL_obj = wf_cl_lib.wig_PyCCL(z_grid, 'with_galaxy_bias', gal_bias_2d_array=gal_bias_2d_array, cosmo=None,
                                    return_PyCCL_object=True)

# ! compute cls
print('computing cls with PyCCL...')
nbl = general_cfg['nbl_WL']
ell_min = general_cfg['ell_min']
ell_max_WL = general_cfg['ell_max_WL']
ell_max_GC = general_cfg['ell_max_GC']
ell_LL, delta_LL = ell_values.compute_ells(nbl=nbl, ell_min=ell_min, ell_max=ell_max_WL, recipe='ISTF')
ell_GG, delta_GG = ell_values.compute_ells(nbl=nbl, ell_min=ell_min, ell_max=ell_max_GC, recipe='ISTF')

# note: I can also pass pk2d=None, which uses the default non-linear pk stored in cosmo. The difference is below 10%.
warnings.warn('I should use pk=None because thats what is used in the derivatives!!!')
cl_LL_3D = wf_cl_lib.cl_PyCCL(wil_PyCCL_obj, wil_PyCCL_obj, ell_LL, zbins, is_auto_spectrum=True, pk2d=None)
cl_GL_3D = wf_cl_lib.cl_PyCCL(wig_PyCCL_obj, wil_PyCCL_obj, ell_GG, zbins, is_auto_spectrum=False, pk2d=None)
cl_GG_3D = wf_cl_lib.cl_PyCCL(wig_PyCCL_obj, wig_PyCCL_obj, ell_GG, zbins, is_auto_spectrum=True, pk2d=None)

# TODO better investigate the pk here
# TODO partial integral? bias outside the kernels?
print('computing cls with Dark...')
start_time = time.perf_counter()
cl_LL_3D_dark = wf_cl_lib.get_cl_3D_array(wil_IA_IST_func, wil_IA_IST_func, ell_LL, is_auto_spectrum=True)
cl_GL_3D_dark = wf_cl_lib.get_cl_3D_array(wig_IST_func, wil_IA_IST_func, ell_GG, is_auto_spectrum=False)
cl_GG_3D_dark = wf_cl_lib.get_cl_3D_array(wig_IST_func, wig_IST_func, ell_GG, is_auto_spectrum=True)
print(f'cl computation took {time.perf_counter() - start_time} seconds with Dark')

cl_output_path = f'{project_path}/output/cl'
cl_benchmarks_path = f'{project_path}/output/cl/benchmarks'
np.save(f'{cl_output_path}/cl_LL_3D.npy', cl_LL_3D)
np.save(f'{cl_output_path}/cl_GL_3D.npy', cl_GL_3D)
np.save(f'{cl_output_path}/cl_GG_3D.npy', cl_GG_3D)
np.save(f'{cl_output_path}/cl_LL_3D_dark.npy', cl_LL_3D_dark)
np.save(f'{cl_output_path}/cl_GL_3D_dark.npy', cl_GL_3D_dark)
np.save(f'{cl_output_path}/cl_GG_3D_dark.npy', cl_GG_3D_dark)
np.save(f'{cl_output_path}/ell_GG.npy', ell_GG)
np.save(f'{cl_output_path}/ell_LL.npy', ell_LL)

# ! a comparison of the cls is in order here!
cl_dict_3D_vinc = mm.load_pickle(f'/Users/davide/Documents/Lavoro/Programmi/cl_v2/data/validation/cl_dict_3D.pickle')
ell_dict_vinc = mm.load_pickle(f'/Users/davide/Documents/Lavoro/Programmi/cl_v2/data/validation/ell_dict.pickle')
delta_dict_vinc = mm.load_pickle(f'/Users/davide/Documents/Lavoro/Programmi/cl_v2/data/validation/delta_dict.pickle')

# ✅ ell values and deltas are the same.

cl_LL_vinc = cl_dict_3D_vinc['cl_LL_3D']
cl_GL_vinc = cl_dict_3D_vinc['cl_3x2pt_5D'][:, 1, 0, :, :]
cl_GG_vinc = cl_dict_3D_vinc['cl_GG_3D']

# np.testing.assert_allclose(cl_LL_3D, cl_LL_vinc, rtol=1e-1, atol=0)
# np.testing.assert_allclose(cl_GL_3D, cl_GL_vinc, rtol=1e-1, atol=0)
np.testing.assert_allclose(cl_GG_3D, cl_GG_vinc, rtol=1e-1, atol=0)

ell_idx = 0
cl_GG_3D = cl_GG_3D[ell_idx, ...]
cl_GG_vinc = cl_GG_vinc[ell_idx, ...]
mm.compare_arrays(cl_GG_3D, cl_GG_vinc, 'cl_GG_3D', 'cl_GG_vinc', plot_array=True, plot_diff=True, log_array=True,
                  plot_diff_threshold=10)

# ! unit test: have the outputs changed?
mm.test_folder_content(cl_output_path, cl_benchmarks_path, 'npy')
mm.test_folder_content(wf_output_path, wf_benchmarks_path, 'npy')

assert 1 > 2

# started to write code for computation of the covmat, but the cls are still in disagreement...
cl_3x2pt_5D = np.zeros((nbl, 2, 2, zbins, zbins))
cl_3x2pt_5D[:, 0, 0, :, :] = cl_LL_3D
cl_3x2pt_5D[:, 0, 1, :, :] = cl_GL_3D.transpose(0, 2, 1)
cl_3x2pt_5D[:, 1, 0, :, :] = cl_GL_3D
cl_3x2pt_5D[:, 1, 1, :, :] = cl_GG_3D

# * compute the covariance matrix
cl_dict_3D = {
    'cl_LL_3D': cl_LL_3D,
    'cl_GL_3D': cl_GL_3D,
    'cl_GG_3D': cl_GG_3D,
    'cl_WA_3D': cl_LL_3D,
    'cl_3x2pt_5D': cl_3x2pt_5D,
}
ell_dict = {
    'ell_WL': ell_LL,
    'ell_GC': ell_GG,
    'ell_WA': ell_LL,  # ! wrong, but I don't use WA at the moment
}
delta_dict = {
    'delta_l_WL': delta_LL,
    'delta_l_GC': delta_GG,
    'delta_l_WA': delta_LL,  # ! wrong, but I don't use WA at the moment
}

assert 1 > 2

ind = mm.build_full_ind(covariance_cfg['triu_tril'], covariance_cfg['row_col_major'], zbins)
covariance_cfg['ind'] = ind

# a random one, again, it will not be used!
Sijkl = np.load('/Users/davide/Documents/Lavoro/Programmi/common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy')

cov_dict = cov_lib.compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D=None,
                               Sijkl=Sijkl)

# time for a little comparison!
cov_pyccl_LL = cov_dict['cov_WL_GO_2D']
cov_dark_LL = np.load('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/ISTF/output/cl15gen/covmat/'
                      'PySSC/covmat_GO_WL_lmax5000_nbl30_zbinsEP10_2D.npz')['arr_0']
del cov_dict
gc.collect()

mm.compare_arrays(cov_pyccl_LL, cov_dark_LL, 'cov_pyccl_LL', 'cov_dark_LL', plot_diff=True, plot_array=True,
                  log_array=True, log_diff=True, plot_diff_threshold=5)

assert 1 > 2

# ! new code: compute the derivatives
fiducial_params = cfg.fiducial_params
free_params = cfg.free_params
fixed_params = cfg.fixed_params
dcl_LL, dcl_GL, dcl_GG = wf_cl_lib.compute_derivatives(fiducial_params, free_params, fixed_params, z_grid, zbins,
                                                       ell_LL, ell_GC, Pk=None)

# TODO output the wf densely sampled to produce covmat with PySSC


print("the script took %.2f seconds to run" % (time.perf_counter() - script_start))
