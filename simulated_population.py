"""
Script for generating a simulated population of Double WDs
with coeval and non-coeval systems.
"""
import pickle
import numpy as np
from mh.MR_relation import logg_from_Teff_M, Teff_from_tau_M
from DWD_class import DWDcontainer
from IFMR_tools import MSLT, IFMR_cls

M_MIN = 1.0 #minimum MS star mass
TEFF_ERR, LOGG_ERR  = 0.02, 0.03
TOT_AGE_MAX = 4.0 #Gyr
ifmr_x_true = np.array([0.5, 2, 4, 8])
ifmr_y_true = np.array([0.15, 0.6, 0.85, 1.4])
IFMR_true = IFMR_cls(ifmr_x_true, ifmr_y_true)
TG_COV = np.diag([20, 20, 0.005, 0.005])
SIGMA_WEIRD = 5.0
N_coeval, N_weird = 40, 10
P_weird_true = N_weird/(N_coeval+N_weird)

def simulated_DWD(outlier=False):
    """
    Create a double white dwarf with its ages and measurements. The primary
    mass, Mi1, is drawn from a Kroupa IMF, with the secondary, Mi2, drawn
    between M_MIN and Mi1. If the outlier argument is False (default) then
    their coeval evolution will be calculated. If True, a normally distributed
    age difference will be added to one of the WDs (SIGMA_WEIRD). Finally,
    measurement uncertainites are added to the Teff and logg measurements.
    """

    #Kroupa IMF P(m) ~ m^-2.3 for m > 0.5
    Mi1 = M_MIN*(np.random.pareto(2.3-1) + 1)
    if Mi1 > 8:
        return None #Mi1 too heavy to become WD

    Mi2 = np.random.uniform(M_MIN, Mi1)

    #50% chance to have B component heaviest
    if np.random.random() < 0.5:
        Mi1, Mi2 = Mi2, Mi1

    #total age of the system
    t_age_tot = np.random.uniform(0, TOT_AGE_MAX)

    t_ms1, t_ms2 = MSLT([Mi1, Mi2])
    t_wd1, t_wd2 = t_age_tot - t_ms1, t_age_tot - t_ms2
    if outlier:
        t_wd1 += np.random.normal(0, SIGMA_WEIRD)
    if any(t < 0 for t in (t_wd1, t_wd2)):
        return None #>=1 star still on MS

    Mf1, Mf2 = IFMR_true([Mi1, Mi2])

    #true Teff and logg
    Teff12 = Teff_from_tau_M([t_wd1, t_wd2], [Mf1, Mf2], 'thick')
    logg12 = logg_from_Teff_M(Teff12, [Mf1, Mf2], 'thick')

    #add measurement uncertainties
    Teff12 *= np.random.normal(1, TEFF_ERR)
    logg12 += np.random.normal(0, LOGG_ERR)

    if not all(5000 < Teff < 25000 for Teff in Teff12):
        return None #Teff too cool for H lines or too hot.

    Tg_vec = *Teff12, *logg12

    if np.any(np.isnan(Tg_vec)):
        return None

    return DWDcontainer("", Tg_vec, TG_COV)

def simulated_DWDs(N_coeval, N_weird):
    """
    Generator function for a population of DWDs with a mixture
    of coeval and outlier systems.
    """
    i_coeval, i_weird = 0, 0
    #Generate coeval DWDs
    while i_coeval < N_coeval:
        DWD = simulated_DWD()
        if DWD is None:
            continue
        DWD.name = f"simulated_coeval_{i_coeval:02}"
        yield DWD
        i_coeval += 1
    #Generate coeval DWDs
    while i_weird < N_weird:
        DWD = simulated_DWD(outlier=True)
        if DWD is None:
            continue
        DWD.name = f"simulated_weird_{i_weird:02}"
        yield DWD
        i_weird += 1

if __name__ == "__main__":
    DWDs = list(simulated_DWDs(N_coeval, N_weird))
    for DWD in DWDs:
        print("{} {:5.0f}K {:5.0f}K {:.3f} {:.3f}".format(DWD.name, *DWD.Tg_vec))
    with open("DWDs_simulated.pkl", 'wb') as F:
        pickle.dump(DWDs, F)
    print("DWDs written to disk")
