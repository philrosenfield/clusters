import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from astropy.table import Table
from match.scripts.ssp import SSP

plt.style.use('presentation')
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)


def interp_for_ov(sspfns):
    for ssp in sspfns:
        target = os.path.split(ssp)[1].split('_')[0]
        figname = '{}_fit_ov.png'.format(target)
        data = pd.read_csv(ssp)
        fig, ax = plt.subplots()
        ovs = [0.3, 0.4, 0.5, 0.6]
        fits = []
        for ov in ovs:
            fit = data['fit'].loc[data['ov'] == ov].copy()
            # prob = np.exp(-0.5 * fit)
            # prob /= prob.sum()
            # fits.append(prob.max())
            # ax.plot(np.repeat(ov, len(fit)), fit, '.')
            fits.append(fit.min())

        z = np.polyfit(ovs, fits, 3)
        x = np.linspace(0.3, 0.6)
        p = np.poly1d(z)
        y = p(x)
        ax.plot(ovs, fits, 'o')
        ax.plot(x, y)
        print(x[np.argmin(y)])
        ax.plot(x[np.argmin(y)], np.min(y), 'o')
        ax.set_title(target)
        plt.savefig(figname)
        plt.close()


def ssp_test(sspfn):
    # (for true9)
    truth = {'IMF': 1.30,
             'dmod': 18.50,
             'Av': 0.1,
             'dav': 0.0,
             'dlogZ': 0.1,
             'bf': 0.0,
             'dmag_min': -1.5,
             'vstep': 0.1,
             'vistep': 0.05,
             'logZ': -0.50,
             'sfr': 5e-3,
             'lage': 10 ** .3127}  # (Msun / yr need to integrate over bin size)
    truelage0 = 10 ** .3000  # Gyr
    truelage1 = 10 ** .3254  # Gyr

    ovs = [0.3, 0.4, 0.5, 0.6]
    ssps = [SSP(sspfn[0], gyr=True, filterby={'trueov': ov})
            for ov in ovs]

    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']

    marg_cols = [c for c in ssps[0].data.columns if c not in avoid_list]
    fignames = ['ssp_test_ov{:.1}.pdf'.format(i) for i in ovs]

    figs, axss, bfs = marg_plots(ssps, marg_cols, truth=truth,
                                 fignames=fignames)

    #lage = [0, truelage0, truelage0, truelage1, truelage1, 0]
    #probage = [0, 0, 1, 1, 0, 0]
    #for i, axs in enumerate(axss):
    #    axs[-1].axvline(ovs[i], color='darkred', lw=3)
    #    axs[3].plot(lage, probage, color='darkred', lw=3)
    #    figs[i].savefig(fignames[i])
    tab = Table.from_pandas(bfs)
    tab.write('best_ssptest.tex')


def cluster_result_plots(sspfns):
    """
    sspfns = ['hodge6_r560_combo.csv',
              'hodge2_r1360_combo.csv',
              'ngc1644_r1000_combo.csv',
              'ngc1718_r1640_combo.csv',
              'ngc1917_r1040_combo.csv',
              'ngc1978_r1600_combo.csv',
              'ngc2173_r1760_combo.csv',
              'ngc2213_r1520_combo.csv',
              'ngc2203_r1520_combo.csv',
              'ngc1795_r1360_combo.csv']
    """
    labelfmt = r'$\rm{{{}}}$'
    avoid_list = ['sfr', 'fit', 'trueov', 'dmag_min', 'vstep',
                  'vistep', 'true', 'tbin', 'ssp', 'dav']
    ssps = []
    labels = []
    marg_cols = []
    fignames = []
    targs = []
    for sspfn in sspfns:
        ssp = SSP(sspfn, gyr=True)
        targ = ssp.name.split('_')[0].upper()

        labels.append(labelfmt.format(targ))
        marg_cols = [c for c in ssp.data.columns if c not in avoid_list]
        fignames.append(sspfn.replace('.csv', '.pdf'))
        ssps.append(ssp)
        targs.append(targs)

    fig, axs, bfs = marg_plots(ssps, marg_cols, text=labels,
                               fignames=fignames)
    bfs['name'] = targs
    tab = Table.from_pandas(bfs)
    tab.write('best_clusters.tex')
    sspfnsb = [s.replace('.csv', '_best.csv') for s in sspfns]
    interp_for_ov(sspfnsb)


def marg_plots(ssps, marg_cols, text=None, truth=None, fignames=None):
    # best fit for each true ov grid:
    if not isinstance(text, list):
        text = [text] * len(ssps)

    if not isinstance(fignames, list):
        fignames = ['{}.pdf'.format(ssp.name) for ssp in ssps]

    bfs = pd.DataFrame()
    figs = []
    axss = []
    for i, ssp in enumerate(ssps):
        if truth is not None:
            truth['ov'] = np.unique(ssp.data['trueov'])
        fig, axs = ssp.pdf_plots(marginals=marg_cols, truth=truth,
                                 text=text[i], twod=True)
        plt.savefig(fignames[i])
        figs.append(fig)
        axss.append(axs)
        bfs = bfs.append(ssp.data.loc[ssp.ibest], ignore_index=True)
    return figs, axss, bfs

if __name__ == "__main__":
    # os.chdir('/Users/rosenfield/research/clusters/asteca/acs_wfc3/paper1')
    ssp_test(sys.argv[1:])
    #cluster_result_plots(sys.argv[1:])
    interp_for_ov(sys.argv[1:])
