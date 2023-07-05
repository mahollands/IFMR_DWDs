#!/usr/bin/env python
import argparse
import numpy as np
import corner
import matplotlib.pyplot as plt
from misc import load_fitted_IFMR

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, default="IFMR_MCMC_230614", nargs='?',
    help="Filename of the IFMR fit (chain or lnprob acceptable)")
parser.add_argument("--burn", type=int, default=100, \
    help="Number of steps to use for MCMC, burning the rest")
parser.add_argument("--thin", type=int, default=5, \
    help="Thinning factor for MCMC steps")
parser.add_argument("--plot_chains", dest="chains", action="store_const", \
    const=True, default=False,
    help="Make figure of MCMC chains")
parser.add_argument("--plot_corner", dest="corner", action="store_const", \
    const=True, default=False,
    help="Make corner plot of IFMR")
parser.add_argument("--corner_hyper", dest="hyper", action="store_const", \
    const=True, default=False,
    help="Make corner plot only show hyperparameters")
parser.add_argument("--plot_ifmr", dest="ifmr", action="store_const", \
    const=True, default=False,
    help="Make IFMR plot")
parser.add_argument("--plot_all", dest="all_figs", action="store_const", \
    const=True, default=False,
    help="Show all chains/corner/ifmr figures")
args = parser.parse_args()

ifmr_x, chain, lnp = load_fitted_IFMR(args.filename)
chain[:,:,2] *= 100

final = chain[:,-args.burn::args.thin,:].reshape((-1, chain.shape[-1]))
best_coords = np.where(lnp == lnp.max())
best = chain[best_coords[0][0], best_coords[1][0]]

Nwalkers, Nstep, Ndim = chain.shape
labels = ["P_outlier", "scale_outlier", "Teff_err", "logg_err"] \
    + [f"$y_{{{x}}}$: ${Mi:.1f}\,M_\odot$" for x, Mi in enumerate(ifmr_x, 1)]

########################################
# Make figures of chains and corner plot

def chain_figure(chain, final, Ndim, Nwalkers):
    plt.figure("chains", figsize=(12, 8))
    for idim, label in enumerate(labels):
        plt.subplot(5, 4, idim+1)
        for iwalker in range(min(Nwalkers, 1000)):
            plt.plot(chain[iwalker,:,idim], 'k-', alpha=0.05)
        plt.plot(np.median(chain[:,:,idim], axis=0), 'r-')
        for pc in (2.3, 15.9, 84.1, 97.7):
            plt.plot(np.percentile(chain[:,:,idim], pc, axis=0), 'C1-')
        plt.ylabel(label)
    plt.tight_layout()
    plt.savefig("figures/IFMR_chains.png", dpi=200)
    plt.show()

def lnprob_figure(lnp, Nwalkers):
    plt.figure("log prob", figsize=(6, 4))
    for iwalker in range(Nwalkers):
        plt.plot(lnp[iwalker,:], 'k-', alpha=0.1)
    plt.ylabel("ln(P)")
    plt.tight_layout()
    plt.show()

def IFMR_figure(final):
    final_ = final[:10000,-len(ifmr_x):]
    best_ = best[-len(ifmr_x):]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for x in ifmr_x:
        plt.axvline(x, c='C3', ls=':', alpha=0.5)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y, 'k-', alpha=0.05)
    for pc in (2.3, 15.9, 84.1, 97.7):
        plt.plot(ifmr_x, np.percentile(final_, pc, axis=0), 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0), 'r-', lw=1.5)
    plt.plot(ifmr_x, best_, '#00FF00', lw=2, ls=':')
    plt.plot([0, 8], [0., 8.], 'b:')
    plt.axhline(1.4, c='r', ls=':')
    plt.xlim(0, 8)
    plt.ylim(0, 2.0)
    plt.xlabel("$M_i$ [M$_\odot$]")
    plt.ylabel("$M_f$ [M$_\odot$]")

    plt.subplot(1, 2, 2)
    for x in ifmr_x:
        plt.axvline(x, c='C3', ls=':', alpha=0.5)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y/ifmr_x, 'k-', alpha=0.05)
    for pc in (2.5, 16, 84, 97.5):
        plt.plot(ifmr_x, np.percentile(final_, pc, axis=0)/ifmr_x, 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0)/ifmr_x, 'r-', lw=1.5)
    plt.plot(ifmr_x, best_/ifmr_x, '#00FF00', lw=2, ls=':')
    plt.xlim(0, 8)
    plt.ylim(0, 1.0)
    plt.xlabel("$M_i$ [M$_\odot$]")
    plt.ylabel("$M_f/M_i$")

    plt.tight_layout()
    plt.savefig("figures/IFMR.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    for idim, label in enumerate(labels):
        pcs = [np.percentile(final[:,idim], pc) for pc in (16, 50, 84)]
        f_args = label, pcs[1], *np.diff(pcs), best[idim]
        print("{} {:.3f}_-{:.3f}^+{:.3f} {:.3f}".format(*f_args))

    if args.chains or args.all_figs:
        chain_figure(chain, final, Ndim, Nwalkers)
        #lnprob_figure(lnp, Nwalkers)

    if args.corner or args.all_figs:
        chain = chain[:,-args.burn::args.thin,:]
        data, labels_ = (chain[:,:,:4], labels[:4]) if args.hyper else (chain, labels)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        corner.corner(data, bins=50, labels=labels_, quantiles=[0.16, 0.50, 0.84], \
            color='#333333')
        plt.savefig("figures/IFMR_corner.png", dpi=200)
        plt.show()

    if args.ifmr or args.all_figs:
        IFMR_figure(final)
