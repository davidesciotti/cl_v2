import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import pyccl as ccl

project_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

sys.path.append(f'{project_path.parent}/SSC_restructured_v2/bin')
import ell_values

import wf_cl_lib

script_start = time.perf_counter()

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# TODO link with config file for stuff like ell_max, ecc

WFs_output_folder = f"WFs_v17"
zbins = 10
zpoints = 1000
z_grid = np.linspace(0, 2.5, zpoints)

"""
# my wf
wig_IST = wf_cl_lib.wig_IST(z_grid, 'with_galaxy_bias')
bias_zgrid = wf_cl_lib.wig_IST(z_grid, 'galaxy_bias_only')

wil_IA_IST = wf_cl_lib.wil_final(z_grid, which_wf='with_IA')
wil_noIA_IST = wf_cl_lib.wil_final(z_grid, which_wf='without_IA')
wil_IAonly_IST = wf_cl_lib.wil_final(z_grid, which_wf='IA_only')

# wf from PyCCL
wil_PyCCL = wf_cl_lib.wil_PyCCL(z_grid, 'with_IA')
wig_PyCCL = wf_cl_lib.wig_PyCCL(z_grid, 'with_galaxy_bias')

# set rainbow colormap over 10 values
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

# check wil
plt.figure()
for i in range(zbins):
    plt.plot(z_grid, wil_PyCCL[:, i], label=f"wil tot i={i}", c=colors[i], ls='-')
    plt.plot(z_grid, wil_IA_IST[:, i], label=f"wil tot i={i}", c=colors[i], ls='-')
plt.legend()
plt.grid()
plt.show()

# check wig
plt.figure()
for i in range(zbins):
    plt.plot(z_grid, wig_IST[:, i], label=f"wig i={i}", c=colors[i], ls='-')
    plt.plot(z_grid, wig_PyCCL[:, i], label=f"wig i={i}", c=colors[i], ls='--')
plt.legend()
plt.grid()
plt.show()

# as well as their "sub-components":
# save everything:
np.save(f'{project_path}/output/WF/{WFs_output_folder}/z_array.npy', z_grid)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/bias_zgrid.npy', bias_zgrid)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/wil_IA_IST_nz{zpoints}.npy', wil_IA_IST)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/wil_noIA_IST_nz{zpoints}.npy', wil_noIA_IST)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/wil_IAonly_IST_nz{zpoints}.npy', wil_IAonly_IST)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/wig_IST_nz{zpoints}.npy', wig_IST)
np.save(f'{project_path}/output/WF/{WFs_output_folder}/wig_nobias_IST_nz{zpoints}.npy', wig_IST.T / bias_zgrid)

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
"""

# Import fiducial P(k,z)
PkFILE = np.genfromtxt(project_path / 'data/pkz-Fiducial.txt')

# ! XXX are the units correct?
# Populate vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]

cosmo = wf_cl_lib.instantiate_PyCCL_cosmology()
zlist = np.unique(PkFILE[:, 0])
k_points = int(len(PkFILE[:, 2]) / len(zlist))
klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
z_points = len(zlist)
Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)  # it's the natural log, not log10
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)


# ! check the cls
print('starting cl computation')
ell_LL, _ = ell_values.compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')
ell_GG, _ = ell_values.compute_ells(nbl=30, ell_min=10, ell_max=3000, recipe='ISTF')
wil_PyCCL_obj = wf_cl_lib.wil_PyCCL(z_grid, 'with_IA', cosmo=None, return_PyCCL_object=True)
wig_PyCCL_obj = wf_cl_lib.wig_PyCCL(z_grid, 'with_galaxy_bias', cosmo=None, return_PyCCL_object=True)
# cl_LL = wf_cl_lib.cl_PyCCL(wil_PyCCL_obj, wil_PyCCL_obj, ell_LL, zbins, is_auto_spectrum=True, pk2d=Pk)
# cl_GL = wf_cl_lib.cl_PyCCL(wig_PyCCL_obj, wil_PyCCL_obj, ell_GG, zbins, is_auto_spectrum=False, pk2d=Pk)
cl_GG = wf_cl_lib.cl_PyCCL(wig_PyCCL_obj, wig_PyCCL_obj, ell_GG, zbins, is_auto_spectrum=True, pk2d=Pk)

i, j = 0, 0
plt.plot(ell_GG, cl_GG[:, 0, 1], label='cl_GG')
plt.plot(ell_GG, cl_GG[:, 1, 0], label='cl_GG')
# plt.plot(ell_LL, cl_LL[:, 0, 0], label='cl_GG')


print("the script took %.2f seconds to run" % (time.perf_counter() - script_start))
