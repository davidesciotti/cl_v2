import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import w0waCDM
from numba import njit

project_path = Path.cwd().parent
sys.path.append(str(project_path))

import config.config as cfg

H0 = cfg.H0
c = cfg.c
Om0 = cfg.Om0
Ode0 = cfg.Ode0
Ox0 = cfg.Ox0
w0 = cfg.w0
wa = cfg.wa
Neff = cfg.Neff
m_nu = cfg.m_nu
Ob0 = cfg.Ob0

cosmo_astropy = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa, Neff=Neff, m_nu=m_nu, Ob0=Ob0)


@njit
def inv_E(z):
    result = 1 / np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
    return result


def E(z):
    return cosmo_astropy.H(z).value / H0


def r(z):
    return cosmo_astropy.comoving_distance(z).value


def r_tilde(z):
    return H0 / c * r(z)

# @njit
# def E(z):
#     result = np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ox0 * (1 + z) ** 2)
#     return result

# old, "manual", slowwwww
# def r_tilde(z):
#     # r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
#     # have r_tilde(z)
#     result = quad(inv_E, 0, z)[0]  # integrate 1/E(z) from 0 to z
#     return result

#
# def r(z):
#     result = c / H0 * quad(inv_E, 0, z)[0]
#     return result
