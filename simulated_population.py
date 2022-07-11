import pickle
import numpy as np
from DWD_class import DWDcontainer
from mh.MR_relation import logg_from_Teff_M, Teff_from_tau_M
from IFMR_tools import MSLT, IFMR_cls

def simulated_DWD_coeval(IFMR, Teff_err, logg_err, outlier=False):
    """
    Simulate a double white dwarf with its ages and measurements
    """
    M_min = 1.0 #shouldn't produce WDs for masses lower than 1Msun

    #Kroupa IMF P(m) ~ m^-2.3 for m > 0.5
    Mi1 = M_min*(np.random.pareto(2.3-1) + 1)
    if Mi1 > 8:
        return None #Mi1 too heavy

    Mi2 = np.random.uniform(M_min, Mi1)

    #50% changce to have B component heaviest
    if np.random.random() < 0.5:
        Mi1, Mi2 = Mi2, Mi1

    t_age_tot = np.random.uniform(0, 4)

    t_ms1, t_ms2 = MSLT([Mi1, Mi2])
    t_wd1, t_wd2 = t_age_tot - t_ms1, t_age_tot - t_ms2
    if any(t < 0 for t in (t_wd1, t_wd2)):
        return None #>=1 star still on MS

    Mf1, Mf2 = IFMR([Mi1, Mi2])

    Teff12 = Teff_from_tau_M([t_wd1, t_wd2], [Mf1, Mf2], 'thick')
    logg12 = logg_from_Teff_M(Teff12, [Mf1, Mf2], 'thick')

    Teff12 *= np.random.normal(1, Teff_err)
    logg12 += np.random.normal(0, logg_err)

    if not all(5000 < Teff < 25000 for Teff in Teff12):
        return None #Teff too cool for H lines or too hot.

    Tg_vec = *Teff12, *logg12
    Tg_cov = np.diag([20, 20, 0.005, 0.005])

    if np.any(np.isnan(Tg_vec)):
        return None

    return DWDcontainer("sim", Tg_vec, Tg_cov) 

def simulated_DWDs(IFMR, Teff_err, logg_err, N_coeval):
    i_coeval = 0
    while True:
        DWD = simulated_DWD_coeval(IFMR, Teff_err, logg_err)
        if DWD is None:
            continue
        DWD.name = f"simulated_coeval_{i_coeval:02}"
        yield DWD
        i_coeval += 1
        if i_coeval == N_coeval:
            break

if __name__ == "__main__":
    ifmr_x = np.array([0.5, 2, 4, 8])
    ifmr_y = np.array([0.15, 0.6, 0.85, 1.4])
    IFMR_true = IFMR_cls(ifmr_x, ifmr_y)

    DWDs = list(simulated_DWDs(IFMR_true, 0.02, 0.03, 50))
    for DWD in DWDs:
        print("{} {:5.0f}K {:5.0f}K {:.3f} {:.3f}".format(DWD.name, *DWD.Tg_vec))

    import matplotlib.pyplot as plt
    loggs = np.ravel([DWD.Tg_vec[2:] for DWD in DWDs])
    plt.hist(loggs, bins=np.arange(7.0, 9.05, 0.1))
    plt.show()

    with open("DWDs_simulated.pkl", 'wb') as F:
        pickle.dump(DWDs, F)
    print("DWDs written to disk")
