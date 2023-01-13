import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

import wf_lib

script_start = time.perf_counter()

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

WFs_output_folder = f"WFs_v17"
zbins = 10
zpoints = 1000
z_grid = np.linspace(0, 4, zpoints)

# let's now test niz
# niz_unnormalized_quadvec_arr = np.asarray([niz_unnormalized_quadvec(z_arr, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_simps_2_arr = np.asarray(
    [wf_lib.niz_unnormalized_simps_2(z_grid, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_simps_arr = np.asarray([wf_lib.niz_unnormalized_simps(z_grid, zbin_idx) for zbin_idx in range(zbins)])
niz_unnormalized_quad_arr = np.asarray([[wf_lib.niz_unnormalized_quad(z, zbin_idx)
                                         for z in z_grid]
                                        for zbin_idx in range(zbins)])
niz_unnormalized_analytical = np.asarray([wf_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
# niz_unnormalized_dav_2 = niz(np.array(range(10)), z_arr).T

# normalize nz: this should be the denominator of Eq. (112) of IST:f
norm_factor_stef = simps(niz_unnormalized_analytical, z_grid)

# niz_normalized_quadvec_arr = normalize_niz(niz_unnormalized_quadvec_arr, z_arr)
niz_normalized_quad_arr = wf_lib.normalize_niz(niz_unnormalized_quad_arr, z_grid)
niz_normalized_simps_arr = wf_lib.normalize_niz(niz_unnormalized_simps_arr, z_grid)
niz_normalized_simps_2_arr = wf_lib.normalize_niz(niz_unnormalized_simps_2_arr, z_grid)
niz_normalized_analytical = wf_lib.normalize_niz(niz_unnormalized_analytical, z_grid)
niz_normalized_cfp = np.load('/Users/davide/Documents/Lavoro/Programmi/cl_v2/input/niz_cosmicfishpie.npy')

# plot them
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for zbin_idx in [0, 5, 9]:
    ax.plot(z_grid, niz_normalized_analytical[zbin_idx], label='analytical', lw=2, ls='-')
    ax.plot(z_grid, niz_normalized_quad_arr[zbin_idx], label='quad', lw=2, ls='--')
    ax.plot(z_grid, niz_normalized_simps_arr[zbin_idx], label='simps', lw=2, ls=':')
    ax.plot(z_grid, niz_normalized_simps_2_arr[zbin_idx], label='simps_2', lw=2)
    # ax.plot(z_arr, niz_normalized_cfp[zbin_idx], label='cfp', lw=1.3)
ax.set_xlabel('z')
ax.set_ylabel('n_i(z)')
ax.legend()
plt.show()

assert 1 > 2

# my wf
wig_IST = wf_lib.wig_IST(z_grid, 'with_galaxy_bias')
bias_zgrid = wf_lib.wig_IST(z_grid, 'galaxy_bias_only')

wil_IA_IST = wf_lib.wil_final(z_grid, which_wf='with_IA')
wil_noIA_IST = wf_lib.wil_final(z_grid, which_wf='without_IA')
wil_IAonly_IST = wf_lib.wil_final(z_grid, which_wf='IA_only')

# wf from PyCCL
wil_PyCCL = wf_lib.wil_PyCCL(z_grid, 'with_IA')
wig_PyCCL = wf_lib.wig_PyCCL(z_grid, 'with_galaxy_bias')

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


print("the script took %.2f seconds to run" % (time.perf_counter() - script_start))
