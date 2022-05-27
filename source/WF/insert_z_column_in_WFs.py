import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
#from functools import partial
#from astropy.cosmology import WMAP9 as cosmo
#from astropy import constants as const
#from astropy import units as u

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"

wil_tot = np.load("%s/output/night_computation/wil_tot.npy" %path)
wig     = np.load("%s/output/night_computation/wig.npy" %path)
wil_SSC = np.load("%s/output/night_computation/wil_SSC.npy" %path)
wig_SSC = np.load("%s/output/night_computation/wig_SSC.npy" %path)
niz     = np.load("%s/output/night_computation/n_i(z).npy" %path)
D       = np.load("%s/output/night_computation/D(z).npy" %path)

z = np.load("%s/output/night_computation/z.npy" %path)

D = np.reshape(D, (z.shape[0], 1))

wil_tot = np.insert(wil_tot, 0, z, axis=1)
wig     = np.insert(wig, 0, z, axis=1)
wil_SSC = np.insert(wil_SSC, 0, z, axis=1)
wig_SSC = np.insert(wig_SSC, 0, z, axis=1)
niz     = np.insert(niz, 0, z, axis=1)
D       = np.insert(D, 0, z, axis=1)

np.save("%s/output/night_computation_zColumn/wil_tot.npy" %path, wil_tot)
np.save("%s/output/night_computation_zColumn/wig.npy" %path, wig)
np.save("%s/output/night_computation_zColumn/wil_SSC.npy" %path, wil_SSC)
np.save("%s/output/night_computation_zColumn/wig_SSC.npy" %path, wig_SSC)
np.save("%s/output/night_computation_zColumn/n_i(z).npy" %path, niz)
np.save("%s/output/night_computation_zColumn/D(z).npy" %path, D)


