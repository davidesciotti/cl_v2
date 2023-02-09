import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent
project_path = '/Users/davide/Documents/Lavoro/Programmi/cl_v2'
project_path_parent = '/Users/davide/Documents/Lavoro/Programmi'


# ! wf settings
use_camb = False  # whether to use camb for the wf (doesn't work yet...)
IA_model = 'eNLA'
useIA = True
zbins = 10
EP_or_ED = 'EP'
flagship_version = 1


load_external_niz = True
# davide
niz_path = f'{project_path}/output/niz'
niz_filename = 'niz_normalized_nz2000.txt'
# flagship
# niz_path = f'{project_path_parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/InputNz/Lenses/Flagship'
# niz_filename = f'niTab-{EP_or_ED}{zbins}.dat'

load_external_bias = False
include_bias = True
bias_selector = 'step_function'  # or "top-hat", or simply "constant?"
bias_path = f'{project_path_parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/InputNz/Lenses/Flagship'
bias_filename = f'ngbTab-{EP_or_ED}{zbins}.dat'

if useIA:
    IA_flag = "IA"
elif not useIA:
    IA_flag = 'noIA'
else:
    raise ValueError('useIA must be True or False')

# ! cl settings
load_external_wf = True
wil_path = f'{project_path_parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/KernelFun'
wil_filename = f'WiWL-{EP_or_ED}{zbins}.dat'
wig_path = wil_path
wig_filename = f'WiGC-{EP_or_ED}{zbins}.dat'

check_plot_wf = False

nbl = 30
ell_min = 10
ell_max_WL = 5000
ell_max_GC = 3000
ell_recipe = 'ISTF'


# xxx is z_max = 4 to be used everywhere?
z_max_cl = 2.9
zsteps_cl = 500  # vincenzo uses 303
units = "1/Mpc"

k_min = 10 ** (-5.442877)
k_max = 30  # in Mpc**-1
k_points = 804

whos_wf = 'marco'
nz_WF_import = 10_000  # number of z points in the wf imported

cl_out_folder = f'cl_v21/Cij_WF{whos_wf}_{IA_flag}_nz{zsteps_cl}'
