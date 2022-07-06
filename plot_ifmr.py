import numpy as np
import corner
import matplotlib.pyplot as plt
from mcmc_functions import MSLT, loglike_Mi12_outliers
from scipy.interpolate import interp1d
from DWD_class import load_DWDs
from misc import create_IFMR

BURN = -50
PLOT_CHAINS = True
PLOT_CORNER = True
PLOT_IFMR = True
PLOT_TOTAL_AGES = True
OUTLIERS = True

if OUTLIERS:
    from fit_ifmr_outliers_errors import ifmr_x, f_MCMC_out
else:
    from fit_ifmr_errors import ifmr_x, f_MCMC_out

chain = np.load(f"MCMC_output/{f_MCMC_out}_chain.npy")
lnp = np.load(f"MCMC_output/{f_MCMC_out}_lnprob.npy")
final = chain[:,-1,:]

Nwalkers, Nstep, Ndim = chain.shape
labels = ["Teff_err", "logg_err"] + [f"y{x}" for x in range(1, len(ifmr_x)+1)]
if OUTLIERS:
    labels = ["P_weird", "scale_weird"] + labels

########################################
# Make figures of chains and corner plot

def chain_figure(chain, final, Ndim, Nwalkers):
    plt.figure("chains", figsize=(12, 8))
    for idim, label in enumerate(labels):
        print(label, np.median(final[:,idim]), np.std(final[:,idim]))
        plt.subplot(5, 3, idim+1)
        for iwalker in range(min(Nwalkers, 1000)):
            plt.plot(chain[iwalker,:,idim], 'k-', alpha=0.05)
        plt.plot(np.median(chain[:,:,idim], axis=0), 'r-')
        for pc in (16, 84):
            plt.plot(np.percentile(chain[:,:,idim], pc, axis=0), 'C1-')
        plt.ylabel(label)
    plt.tight_layout()
    plt.savefig("IFMR_chains.png", dpi=200)
    plt.show()

def lnprob_figure(lnp, Nwalkers):
    plt.figure("log prob", figsize=(6, 4))
    for iwalker in range(Nwalkers):
        plt.plot(lnp[iwalker,:], 'k-', alpha=0.1)
    plt.ylabel("ln(P)")
    plt.tight_layout()
    plt.show()

def IFMR_figure(final):

    final_ = final[:,-len(ifmr_x):]
    
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y, 'k-', alpha=0.05)
    plt.plot(ifmr_x, np.percentile(final_, 16, axis=0), 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0), 'r-')
    plt.plot(ifmr_x, np.percentile(final_, 84, axis=0), 'C1-')
    plt.plot([0, 8], [0., 8.], 'b:')
    plt.axhline(1.4, c='r', ls=':')
    plt.xlim(0, 8)
    plt.ylim(0, 2.0)
    plt.xlabel("Mi [Msun]")
    plt.ylabel("Mf [Msun]")

    plt.subplot(1, 2, 2)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y/ifmr_x, 'k-', alpha=0.05)
    plt.plot(ifmr_x, np.percentile(final_, 16, axis=0)/ifmr_x, 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0)/ifmr_x, 'r-')
    plt.plot(ifmr_x, np.percentile(final_, 84, axis=0)/ifmr_x, 'C1-')
    plt.xlim(0, 8)
    plt.ylim(0, 1.0)
    plt.xlabel("Mi [Msun]")
    plt.ylabel("Mf/Mi")

    plt.tight_layout()
    plt.savefig("IFMR.png", dpi=200)
    plt.show()

def total_ages_figure(final, DWD):
    plt.figure(DWD.name, figsize=(6, 6))
    plt.plot([0,15], [0,15], 'k:')
    P_coeval = []
    for params in final:
        if OUTLIERS:
            P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = params
        else:
            Teff_err, logg_err, *ifmr_y = params
        IFMR = create_IFMR(ifmr_x, ifmr_y)
        covMtau = DWD.covMtau_systematics(Teff_err, logg_err)

        Mf1, Mf2, tau1, tau2 = np.random.multivariate_normal(DWD.vecMtau, covMtau)
        if not (ifmr_y[0] < Mf1 < ifmr_y[-1] and ifmr_y[0] < Mf2 < ifmr_y[-1]):
            continue
        Mi1, Mi2 = IFMR.inv([Mf1, Mf2])

        Mi12 = np.array([Mi1, Mi2])

        if OUTLIERS:
            if not (0.6 < Mi1 < 8 and 0.6 < Mf2 < 8):
                P_i = 0
            else:
                covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
                logL_coeval, logL_weird = loglike_Mi12_outliers(Mi12, \
                    DWD.vecMdtau, covMdtau, IFMR, P_weird, scale_weird, separate=True)
                logL_tot = np.logaddexp(logL_coeval, logL_weird)
                P_i = float(np.exp(logL_coeval-logL_tot))
            P_coeval.append(P_i)

        t1, t2 = MSLT(Mi1), MSLT(Mi2)
        if OUTLIERS:
            plt.scatter(t1+tau1, t2+tau2, s=1, c=P_i, vmin=0, vmax=1, )
        else:
            plt.plot(t1+tau1, t2+tau2, 'C0.', ms=1)

    P_coeval = np.array(P_coeval)
    print(DWD.name, np.mean(P_coeval), np.median(P_coeval))

    plt.loglog()
    plt.xlim(0.2, 13)
    plt.ylim(0.2, 13)
    tick_labels = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    plt.xticks(tick_labels, tick_labels)
    plt.yticks(tick_labels, tick_labels)
    plt.xlabel("t_tot1 [Gyr]")
    plt.ylabel("t_tot2 [Gyr]")
    if OUTLIERS:
        plt.title("$P_\mathrm{{coeval}}$ = {:.3f}".format(np.mean(P_coeval)))
    plt.tight_layout()
    plt.savefig(f"age_plots/{DWD.name}.png", dpi=200)
    plt.close()
    #plt.show()

if __name__ == "__main__":
    if PLOT_CHAINS:
        chain_figure(chain, final, Ndim, Nwalkers)
        #lnprob_figure(lnp, Nwalkers)

    if PLOT_CORNER:
        data = chain[:,BURN::5,:]
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        corner.corner(data, smooth1d=True, labels=labels, quantiles=[0.16, 0.50, 0.84])
        plt.savefig("IFMR_corner.png", dpi=200)
        plt.show()

    if PLOT_IFMR:
        IFMR_figure(final)

    if PLOT_TOTAL_AGES:
        DWDs = load_DWDs()
        for DWD in DWDs:
            total_ages_figure(final, DWD)
