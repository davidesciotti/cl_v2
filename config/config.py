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


# ! wf settings
use_camb = False  # whether to use camb for the wf (doesn't work yet...)
IA_model = 'eNLA'
useIA = True

load_external_niz = False
niz_path = f'{project_path}/input/niz'
niz_filename = 'niz.txt'

if useIA:
    IA_flag = "IA"
elif not useIA:
    IA_flag = 'noIA'
else:
    raise ValueError('useIA must be True or False')

# ! cl settings
nbl = 30
ell_min = 10
ell_max_WL = 5000
ell_max_GC = 3000

bias_selector = 'newBias'

# xxx is z_max = 4 to be used everywhere?
z_max_cl = 4.
zsteps_cl = 500  # vincenzo uses 303
units = "1/Mpc"

k_min = 10 ** (-5.442877)
k_max = 30  # in Mpc**-1
k_points = 804

whos_wf = 'marco'
nz_WF_import = 10_000  # number of z points in the wf imported

cl_out_folder = f'cl_v21/Cij_WF{whos_wf}_{IA_flag}_nz{zsteps_cl}'
