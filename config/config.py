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

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
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
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

IA_model = 'eNLA'
useIA = True

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
