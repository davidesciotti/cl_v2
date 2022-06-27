import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import w0waCDM
from classy import Class
from numba import njit

project_path = Path.cwd().parent
sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))

import ISTF_fid_params as ISTF
import config.config as cfg

c = ISTF.constants['c']

Om0 = ISTF.primary['Om_m0']
Ob0 = ISTF.primary['Om_b0']
w0 = ISTF.primary['w_0']
wa = ISTF.primary['w_a']
h = ISTF.primary['h_0']
n_s = ISTF.primary['n_s']
sigma_8 = ISTF.primary['sigma_8']

Ode0 = ISTF.extensions['Om_Lambda0']
Ok0 = ISTF.extensions['Om_k0']

Neff = ISTF.neutrino_params['N_eff']
m_nu = ISTF.extensions['m_nu']

Oc0 = Om0 - Ob0
H0 = h * 100

cosmo_astropy = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa, Neff=Neff, m_nu=m_nu, Ob0=Ob0)

cosmo_par_dict = {'Omega_b': Ob0,
                  'Omega_cdm': Oc0,
                  'n_s': n_s,
                  'sigma8': sigma_8,
                  'h': h,
                  'output': 'mPk',
                  'z_pk': '0, 0.5, 1, 2, 3',
                  'P_k_max_h/Mpc': 50,
                  'non linear': 'halofit'}

cosmo_classy = Class()
cosmo_classy.set(cosmo_par_dict)
cosmo_classy.compute()


@njit
def inv_E(z):
    result = 1 / np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ok0 * (1 + z) ** 2)
    return result


def E(z):
    return cosmo_astropy.H(z).value / H0


def r(z):
    return cosmo_astropy.comoving_distance(z).value


def r_tilde(z):
    return H0 / c * r(z)


def k_limber(ell, z):
    return (ell + 1 / 2) / r(z)


# @njit
# def E(z):
#     result = np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ok0 * (1 + z) ** 2)
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


def Pk_with_classy_clustertlkt(cosmo, z_array, k_array, use_h_units, Pk_kind='nonlinear', argument_type='arrays'):
    print('Warning: this function takes as input k in 1/Mpc and returns it in the specified units')

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo.pk_lin
    else:
        raise ValueError('Pk_kind must be either "nonlinear" or "linear"')

    if argument_type == 'scalar':
        Pk = classy_Pk(k_array, z_array)  # k_array and z_array are not arrays, but scalars!

    elif argument_type == 'arrays':
        num_k = k_array.size

        Pk = np.zeros((len(z_array), num_k))
        for z_idx, z_val in enumerate(z_array):
            Pk[z_idx, :] = np.array([classy_Pk(ki, z_val) for ki in k_array])

    # NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
    # to use in the toolkit. To do this you would do:
    if use_h_units:
        k_array /= h
        Pk *= h ** 3

    # return also k_array, to have it in the correct h scaling
    return k_array, Pk


def calculate_power(cosmo, z_array, k_array, use_h_units=True, Pk_kind='nonlinear', argument_type='arrays'):
    if use_h_units:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo.pk_lin
    else:
        raise ValueError('Pk_kind must be either "nonlinear" or "linear"')

    # if z and k are not arrays, return scalar output
    if argument_type == 'scalar':
        Pk = classy_Pk(k_array * k_scale, z_array) * Pk_scale  # k_array and z_array are not arrays, but scalars!

    elif argument_type == 'arrays':
        num_k = k_array.size

        Pk = np.zeros((len(z_array), num_k))
        for zi, z in enumerate(z_array):
            for ki, k in enumerate(k_array):
                # the argument of classy_Pk must be in units of 1/Mpc?
                Pk[zi, ki] = classy_Pk(k * k_scale, z) * Pk_scale

    else:
        raise ValueError('argument_type must be either "scalar" or "arrays"')

    return Pk


def get_external_Pk(whos_Pk='vincenzo', Pk_kind='nonlinear', use_h_units=True):

    assert type(use_h_units) == bool, 'use_h_units must be True or False'

    if whos_Pk == 'vincenzo':
        filename = 'PnlFid.dat'
        z_column = 1
        k_column = 0  # in [1/Mpc]
        Pnl_column = 2  # in [Mpc^3]
        Plin_column = 3  # in [Mpc^3]

    elif whos_Pk == 'stefano':
        filename = 'pkz-Fiducial.txt'
        z_column = 0
        k_column = 1  # in [h/Mpc]
        Pnl_column = 3  # in [Mpc^3/h^3]
        Plin_column = 2  # in [Mpc^3/h^3]

    if Pk_kind == 'linear':
        Pk_column = Plin_column
    elif Pk_kind == 'nonlinear':
        Pk_column = Pnl_column
    else:
        raise ValueError(f'Pk_kind must be either "linear" or "nonlinear"')

    Pkfile = np.genfromtxt(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/input/variable_response/{filename}')
    z_array = np.unique(Pkfile[:, z_column])
    k_array = np.unique(Pkfile[:, k_column])
    Pk = Pkfile[:, Pk_column].reshape(z_array.size, k_array.size)  # / h ** 3

    if whos_Pk == 'vincenzo':
        k_array = 10 ** k_array
        Pk = 10 ** Pk

    # h scaling
    if use_h_units is True:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
        if whos_Pk == 'vincenzo':
            k_array /= h
            Pk *= h ** 3
    elif use_h_units is False:
        if whos_Pk == 'stefano':  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
            k_array *= h
            Pk /= h ** 3

    # flip, the redshift array is ordered from high to low
    Pk = np.flip(Pk, axis=0)

    return z_array, k_array, Pk


def k_limber(z, ell, cosmo_astropy, use_h_units):

    assert type(use_h_units) == bool, 'use_h_units must be True or False'

    # astropy gives values in Mpc, so I call astropy_comoving_distance to have the correct values in both cases
    comoving_distance = astropy_comoving_distance(z, cosmo_astropy, use_h_units)
    k_ell = (ell + 0.5) / comoving_distance
    return k_ell

def k_limber_new(z, ell, cosmo_astropy, use_h_units):

    """ I get an error if ell is a vector! Let's try to correct that:"""

    assert type(use_h_units) == bool, 'use_h_units must be True or False'

    # astropy gives values in Mpc, so I call astropy_comoving_distance to have the correct values in both cases
    comoving_distance = astropy_comoving_distance(z, cosmo_astropy, use_h_units)
    k_ell = (ell + 0.5) / comoving_distance
    return k_ell


def astropy_comoving_distance(z, cosmo_astropy, use_h_units):
    if use_h_units:
        return cosmo_astropy.comoving_distance(z).value * h  # Mpc/h
    else:
        return cosmo_astropy.comoving_distance(z).value  # Mpc
