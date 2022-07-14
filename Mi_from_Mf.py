"""
Tool for converting final masses to initial masses, including uncertainies
in the final mass and the IFMR itself.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IFMR_tools import IFMR_cls, MSLT

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
if args.Mf_mean < 0:
    raise ValueError("Mf err must be positive")

if args.outliers:
    from fit_ifmr_outliers_errors import ifmr_x, f_MCMC_out
else:
    from fit_ifmr_errors import ifmr_x, f_MCMC_out

chain = np.load(f"MCMC_output/{f_MCMC_out}_chain.npy")
final = chain[:,-1,:]

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
    t_pcs *= 1000
    t_unit = "Myr"
    t_fmt = "{:.0f}_-{:.0f}^+{:.0f}"
else:
    t_unit = "Gyr"
    t_fmt = "{:.2f}_-{:.2f}^+{:.2f}"
t_errs = np.diff(t_pcs)
t_str = t_fmt.format(t_pcs[1], *t_errs)

print(f"Mf : {Mf_str} Msun ==> Mi : {Mi_str} Msun, MS_t = {t_str} {t_unit}")

if args.show:
    plt.hist(Mi, bins=np.arange(0, 8, 0.1), density=True, histtype='step')
    for Mi_pc in Mi_pcs:
        plt.axvline(Mi_pc, c='C1', ls=':')
    plt.xlabel("$M_i [M_\odot]$")
    plt.ylabel("$P(M_i)$")
    plt.xlim(0, 8)
    plt.show()
