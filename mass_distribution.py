import numpy as np
import matplotlib.pyplot as plt
from DWD_class import load_DWDs
from DWD_sets import bad_DWDs_220809 as bad_DWDs

DWDs = load_DWDs()
DWDs_fit = [DWD for DWD in DWDs if not DWD.name in bad_DWDs]

Masses_all = [float(M) for DWD in DWDs for M in DWD.vecMtau[:2]]
Masses_fit = [float(M) for DWD in DWDs_fit for M in DWD.vecMtau[:2]]

plt.hist(Masses_all, bins=np.arange(0, 1.4, 0.05), histtype='step')
plt.hist(Masses_fit, bins=np.arange(0, 1.4, 0.05), histtype='step', ls='--')
plt.xlabel("$M_f [M_\odot]$")
plt.ylabel("Number per bin")
plt.show()

for DWD in DWDs:
    T, g = DWD.Tg_vec[:2], DWD.Tg_vec[2:]
    err = np.sqrt(np.diag(DWD.Tg_cov))
    Terr, gerr = err[:2], err[2:]
    c = 'C0' if DWD.name in bad_DWDs else 'C1'
    plt.errorbar(T, g, gerr, Terr, fmt=c+'.')
plt.xlabel(r"$T_\mathrm{eff}$ [K]")
plt.ylabel(r"$\log g$")
plt.show()
