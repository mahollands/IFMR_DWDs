#!/usr/bin/env python
"""
Tool for converting final masses to initial masses, including uncertainies
in the final mass and the IFMR itself.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IFMR_tools import IFMR_cls, MSLT
from misc import load_fitted_IFMR

f_MCMC_out = "IFMR_MCMC_230605"
BURN = -100
THIN = 5

parser = argparse.ArgumentParser()
parser.add_argument("Mf_mean", type=float, \
    help = "Central value of the white dwarf final mass in Msun")
parser.add_argument("Mf_err", type=float, \
    help = "Uncertainty on the white dwarf final mass in Msun")
parser.add_argument("--no_outliers", dest="outliers", action="store_const", \
    const=False, default=True, help = "Don't use outliers solution")
parser.add_argument("--show", action="store_const", \
    const=True, default=False, help = "Show figure of Mi distribution")
args = parser.parse_args()

if not 0 < args.Mf_mean < 1.4:
    raise ValueError("Mf mean should be valid WD mass")

ifmr_x, chain, lnp = load_fitted_IFMR(f_MCMC_out)
final = chain[:,BURN::THIN,:].reshape((-1,chain.shape[-1]))

Nskip = 4 if args.outliers else 2
IFMRs = [IFMR_cls(ifmr_x, ifmr_y) for ifmr_y in final[:,Nskip:]]

Mf = np.random.normal(args.Mf_mean, args.Mf_err, len(IFMRs))
Mi = np.array([IFMR.inv(mf) for IFMR, mf in zip(IFMRs, Mf) \
    if IFMR.y[0] < mf < IFMR.y[-1]])

Mi_pcs = [np.percentile(Mi, pc) for pc in (15.9, 50, 84.1)]
Mi_errs = np.diff(Mi_pcs)

if args.Mf_err > 0:
    Mf_str = "{:.2f}±{:.2f}".format(args.Mf_mean, args.Mf_err)
else:
    Mf_str = "{:.2f}".format(args.Mf_mean)
Mi_str = "{:.2f}_-{:.2f}^+{:.2f}".format(Mi_pcs[1], *Mi_errs)

t = MSLT(Mi)
t_pcs = np.array([np.percentile(t, pc) for pc in (15.9, 50, 84.1)])
if t_pcs[1] < 1:
    t *= 1000
    t_pcs *= 1000
    t_unit = "Myr"
    t_fmt = "{:.0f}_-{:.0f}^+{:.0f}"
else:
    t_unit = "Gyr"
    t_fmt = "{:.2f}_-{:.2f}^+{:.2f}"
t_errs = np.diff(t_pcs)
t_str = t_fmt.format(t_pcs[1], *t_errs)

print(f"Mf : {Mf_str} Msun ==> Mi : {Mi_str} Msun, t_preWD = {t_str} {t_unit}")

if args.show:
    plt.hist(Mi, bins=np.arange(0, 8, 0.05), density=True, histtype='step')
    for Mi_pc in Mi_pcs:
        plt.axvline(Mi_pc, c='C1', ls=':')
    plt.xlabel(r"$M_i [M_\odot]$")
    plt.ylabel(r"$P(M_i)$")
    plt.xlim(0, 8)
    plt.show()

    plt.hist(t, bins=100, density=True, histtype='step')
    for t_pc in t_pcs:
        plt.axvline(t_pc, c='C1', ls=':')
    plt.xlabel(rf"$\tau_\mathrm{{preWD}}$ [{t_unit}]")
    plt.ylabel(r"$P(\tau_\mathrm{preWD})$")
    plt.show()
