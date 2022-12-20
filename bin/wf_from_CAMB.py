import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

matplotlib.use('TkAgg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! new code - just a test with CAMB WF
################# CAMB #####################
import camb
from camb import model, initialpower

Om_c0 = ISTF.primary['Om_m0'] - ISTF.primary['Om_b0'] - ISTF.neutrino_params['Omega_nu']
omch2 = Om_c0 * ISTF.primary['h_0'] ** 2

# Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=H0, ombh2=ISTF.primary['Om_bh2'], omch2=omch2,
                   mnu=ISTF.extensions['m_nu'], omk=ISTF.extensions['Om_k0'], tau=ISTF.other_cosmo_params['tau'])

pars.InitPower.set_params(As=ISTF.other_cosmo_params['A_s'], ns=ISTF.primary['n_s'], r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

# start with one bin
# wil = camb.sources.SplinedSourceWindow(source_type = 'lensing')

pars.SourceWindows = [
    GaussianSourceWindow(redshift=0.001, source_type='counts', bias=b(0), sigma=0.04, dlog10Ndm=-0.2),
    GaussianSourceWindow(redshift=0.5, source_type='lensing', sigma=0.07)]

results = camb.get_results(pars)
cls = results.get_source_cls_dict()

# import vincenzo:
path_vinc = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Cij-NonLin-eNLA_15gen'
cl_vinc = np.genfromtxt(f'{path_vinc}/CijLL-LCDM-NonLin-eNLA.dat')

lmax = 2500
ls = np.arange(2, lmax + 1)
# for spectrum in ['W1xW1', 'W2xW2', 'W1xW2']:
for spectrum in ['W1xW1']:
    plt.loglog(ls, cls[spectrum][2: lmax + 1] * 2 * np.pi / (ls * (ls + 1)), label=spectrum)
    plt.plot(cl_vinc[:, 0], cl_vinc[:, 1])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi$')
plt.legend()