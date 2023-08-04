#!/usr/bin/env python
"""
Tool for converting final masses to initial masses, including uncertainies
in the final mass and the IFMR itself.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IFMR_tools import IFMR_cls, MSLT
from misc import load_fitted_IFMR

parser = argparse.ArgumentParser()
parser.add_argument("Mf_mean", type=float, \
    help = "Central value of the white dwarf final mass in Msun")
parser.add_argument("Mf_err", type=float, \
    help = "Uncertainty on the white dwarf final mass in Msun")
parser.add_argument("--IFMR", type=str, default="IFMR_MCMC_Fit1", \
    help = "Name of the IFMR file to load")
parser.add_argument("--clip_l", type=int, default=0, \
    help = "Number of IFMR points to clip from the left")
parser.add_argument("--clip_r", type=int, default=0, \
    help = "Number of IFMR points to clip from the right")
parser.add_argument("--no_outliers", dest="outliers", action="store_const", \
    const=False, default=True, help = "Don't use outliers solution")
parser.add_argument("--show", action="store_const", \
    const=True, default=False, help = "Show figures of Mi and tau_preWD distributions")
args = parser.parse_args()

if not 0 < args.Mf_mean < 1.4:
    raise ValueError("Mf mean should be valid WD mass")

Nskip = 4 if args.outliers else 2
ifmr_x, chain, lnp = load_fitted_IFMR(args.IFMR)
ifmr_y_ = chain[:,-1:,Nskip:].reshape((-1, len(ifmr_x)))

IFMRs = [IFMR_cls(ifmr_x, ifmr_y) for ifmr_y in ifmr_y_]
if args.clip_l > 0:
    IFMRs = [IFMR[args.clip_l:] for IFMR in IFMRs]
if args.clip_r > 0:
    IFMRs = [IFMR[:-args.clip_r] for IFMR in IFMRs]

dMi = 0.001
Mi_grid = np.arange(0.5, 8+dMi/2, dMi)
prior = Mi_grid**-2.3 #Kroupa IMF for m>0.5Msun
likes = (stats.norm.pdf(IFMR(Mi_grid), 
    loc=args.Mf_mean, scale=args.Mf_err) for IFMR in IFMRs)
Ps = (prior * like for like in likes)
P = sum(p/p.sum() for p in Ps)/len(IFMRs)
Mi = np.random.choice(Mi_grid, size=10_000, p=P)

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
    plt.plot(Mi_grid, P/np.diff(Mi_grid).mean())
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
