import argparse
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
             'lage': 10 ** .3127}  # (Msun/yr need to integrate over bin size)
    truelage0 = 10 ** .3000  # Gyr
    truelage1 = 10 ** .3254  # Gyr

    ovs = [0.3, 0.4, 0.5, 0.6]
    ssps = [SSP(sspfn[0], gyr=True, filterby={'trueov': ov})
            for ov in ovs]

    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']

    marg_cols = [c for c in ssps[0].data.columns if c not in avoid_list]
    fignames = ['ssp_test_ov{:.1}.pdf'.format(i) for i in ovs]

    for i, ssp in enumerate(ssps):
        if truth is not None:
            truth['ov'] = np.unique(ssp.data['trueov'])
        fig, axs = ssp.pdf_plots(marginals=marg_cols, truth=truth,
                                 text=text[i], twod=True, cmap=plt.cm.Reds)
        plt.savefig(fignames[i])
    return


def cluster_result_plots(sspfns, oned=False, twod=False, onefig=False):
    """corner plot of a big combine scrn output"""
    labelfmt = r'$\rm{{{}}}$'
    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']

    # This assures the same order on the plots, though they are default in ssp
    # marg_cols = ['Av', 'bf', 'dmod', 'lage', 'logZ', 'ov']
    marg_cols = ['Av', 'bf', 'dmod', 'lage', 'logZ', 'vvcrit']
    frompost = False
    nssps = len(sspfns)
    ndim = len(marg_cols)
    fig = None
    axs = [None] * nssps

    if onefig:
        fig, axs = plt.subplots(nrows=nssps, ncols=ndim,
                                figsize=(ndim * 1.4, nssps))

    for i, sspfn in enumerate(sspfns):
        print(sspfn)
        if sspfn.endswith('.csv'):
            # ssp = SSP(sspfn, gyr=True)
            ssp = SSP(sspfn)
            # ssp.gyr = False
            # Checks for more than one unique value to marginalize over
            # also adds unique arrays as attributes so this won't add
            # computation time.
            ssp.check_grid(skip_cols=avoid_list)
            # import pdb; pdb.set_trace()
        else:
            ssp = SSP()
            ssp.gyr = True
            ssp.load_posterior(sspfn)
            frompost = True
        targ = ssp.name.split('_')[0].upper()
        label = labelfmt.format(targ)
        ylabel = labelfmt.format(targ)
        if onefig:
            label = None

        if oned:
            f, raxs = ssp.pdf_plots(marginals=marg_cols, text=label,
                                    fig=fig, axs=axs[i], frompost=frompost)

            if not onefig:
                figname = sspfn.replace('.csv', '_1d.pdf')
                plt.savefig(figname)
                plt.close()
            else:
                raxs[-1].set_ylabel(ylabel)
                raxs[-1].yaxis.set_label_position("right")

            if not frompost:
                ssp.write_posterior(filename='{}_post.dat'.format(targ))
        if twod:
            ssp.pdf_plots(marginals=marg_cols, text=label, twod=True,
                          cmap=plt.cm.Reds)
            figname = sspfn.replace('.csv', '.pdf')
            plt.savefig(figname)
            plt.close()
    if onefig:
        for ax in axs.ravel()[:-1*ndim]:
            ax.tick_params(labelbottom='off', tickdir='in')
            ax.axes.set_xlabel('')
        [ax.axes.set_ylabel('') for ax in axs[:, 0]]
        fig.text(0.02, 0.5, labelfmt.format('Probability'), ha='center',
                 va='center', rotation='vertical')
        fig.subplots_adjust(hspace=0.07, wspace=0.07, right=0.95, top=0.98,
                            bottom=0.08)
        figname = 'combo_{}_ssps.pdf'.format(nssps)
        plt.savefig(figname)
        plt.close()
    return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="pdf plots for a odyssey calcsfh run")

    parser.add_argument('-t', '--test', action='store_true',
                        help='ssp test')

    parser.add_argument('--oned', action='store_true',
                        help='1d pdf plots')

    parser.add_argument('--twod', action='store_true',
                        help='corner plots')

    parser.add_argument('--onefig', action='store_true',
                        help='with --oned put all csv files on one plot')

    parser.add_argument('filename', type=str, nargs='*', help='csv file(s)')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.test:
        ssp_test(args.filename)
    else:
        cluster_result_plots(args.filename, oned=args.oned, twod=args.twod,
                             onefig=args.onefig)

if __name__ == "__main__":
    sys.exit(main())
