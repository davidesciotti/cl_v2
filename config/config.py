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


z_minus = np.array((0.0010, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58))
z_plus = np.array((0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50))



# xxx is z_max = 4 to be used everywhere?
z_max = 4




# cl settings
nbl = 30
whos_wf = 'vincenzo'
znum_cl = 500
units = "1/Mpc"


