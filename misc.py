import numpy as np
from mh.MR_relation import M_from_Teff_logg, tau_from_Teff_logg

def pairwise(collection):
    """
    Roughly recreates the python 3.10 itertools.pairwise. Will only work for
    collections, not iterators.
    """
    return zip(collection, collection[1:])

def Taylor_Expand_WD(T0, g0, dT=10, dg=0.001, thickness='thick'):
    """
    Pre-compute Taylor-Expansions for Masses and ages
    """
    T_vals = np.array([T0, T0+dT, T0-dT, T0, T0])
    g_vals = np.array([g0, g0, g0, g0+dg, g0-dg])

    M_vals = M_from_Teff_logg(T_vals, g_vals, thickness)
    M0, M_p_dT , M_m_dT , M_p_dg , M_m_dg = M_vals
    dM_dT = (M_p_dT - M_m_dT)/(2*dT)
    dM_dg = (M_p_dg - M_m_dg)/(2*dg)
    Taylor_M = M0, dM_dT, dM_dg

    tau_vals = tau_from_Teff_logg(T_vals, g_vals, thickness)
    tau0, tau_p_dT, tau_m_dT, tau_p_dg, tau_m_dg = tau_vals
    dtau_dT = (tau_p_dT - tau_m_dT)/(2*dT)
    dtau_dg = (tau_p_dg - tau_m_dg)/(2*dg)
    Taylor_tau = tau0, dtau_dT, dtau_dg

    return Taylor_M, Taylor_tau

def Taylor_Expand_DWD(DWD, separate_tau=False):
    (T1, T2, g1, g2), *_ = DWD
    (M10, dM1_dT1, dM1_dg1), (tau10, dtau1_dT1, dtau1_dg1) = Taylor_Expand_WD(T1, g1)
    (M20, dM2_dT2, dM2_dg2), (tau20, dtau2_dT2, dtau2_dg2) = Taylor_Expand_WD(T2, g2)
    if separate_tau:
        Jac = np.array([
            [dM1_dT1, 0, dM1_dg1, 0],
            [0, dM2_dT2, 0, dM2_dg2],
            [dtau1_dT1, 0, dtau1_dg1, 0],
            [0, dtau2_dT2, 0, dtau2_dg2],
        ])
        vec = np.array([M10, M20, tau10, tau20])
    else:
        Jac = np.array([
            [dM1_dT1, 0, dM1_dg1, 0],
            [0, dM2_dT2, 0, dM2_dg2],
            [dtau1_dT1, -dtau2_dT2, dtau1_dg1, -dtau2_dg2],
        ])
        vec = np.array([M10, M20, tau10-tau20])

    return *DWD, vec, Jac
