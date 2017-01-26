
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


def bycov(sspfns, oned=True, twod=False, onefig=False):
        labelfmt = r'$\rm{{{}}}$'
        avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                      'trueov', 'dav', 'ov']

        # This assures the same order on the plots, though they are default in ssp
        marg_cols = ['Av', 'bf', 'dmod', 'lage', 'logZ']
        ovs = [0.3, 0.4, 0.5, 0.6]
        # marg_cols = ['Av', 'bf', 'dmod', 'lage', 'logZ', 'vvcrit']
        ndim = len(marg_cols)
        nrows = len(ovs)
        for sspfn in sspfns:
            print(sspfn)
            fig = None
            axs = [None] * len(ovs)
            if onefig:
                fig, axs = plt.subplots(nrows=nrows, ncols=ndim,
                                        figsize=(ndim * 1.4, nrows))

            for i, ov in enumerate(ovs):
                ssp = SSP(sspfn, gyr=True, filterby={'IMF': 1.35,
                                                     'ov': ov})
                ssp.check_grid(skip_cols=avoid_list)

                targ = ssp.name.split('_')[0].upper()
                label = ''# r'\rm{{{0:s}}}\ \Lambda_\rm{{c}}={1:.1f}'.format(targ, ov)

                if oned:
                    f, raxs = ssp.pdf_plots(marginals=marg_cols, text=label,
                                            fig=fig, axs=axs[i])
                if not onefig:
                    figname = sspfn.replace('.csv', 'ov{0:.1f}_1d.pdf'.foramt(ov))
                    plt.savefig(figname)
                    plt.close()
                else:
                    ylabel = r'\Lambda_c={}'.format(ov)
                    raxs[-1].set_ylabel(ylabel)
                    raxs[-1].yaxis.set_label_position("right")


                ssp.write_posterior(filename='{0:s}_ov{1:.1f}_post.dat'.format(targ, ov))
                if twod:
                    ssp.pdf_plots(marginals=marg_cols, text=label, twod=True,
                                  cmap=plt.cm.Reds)
                    figname = sspfn.replace('.csv', '_ov{0:.1f}.pdf'.format(ov))
                    plt.savefig(figname)
                    plt.close()
            if onefig:
                import pdb; pdb.set_trace()
                fig, axs = fixcorner(fig, axs, ndim)
                figname = '{}_{}_ssps_ov.pdf'.format(targ, nrows)
                plt.savefig(figname)
                plt.close()
        return


def cluster_result_plots(sspfns, oned=False, twod=False, onefig=False,
                         mist=False):
    """corner plot of a big combine scrn output"""
    labelfmt = r'$\rm{{{}}}$'
    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']
    if mist:
        avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'ssp',
                      'trueov', 'dav']

    # This assures the same order on the plots, though they are default in ssp
    marg_cols = ['Av', 'dmod', 'lage', 'logZ', 'ov']
    if mist:
        marg_cols = ['Av', 'dmod', 'lage', 'logZ', 'vvcrit', 'tbin']
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
            ssp = SSP(sspfn, gyr=True)# , filterby={'IMF': 1.35, 'bf': 0.3})
            # ssp = SSP(sspfn)
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
        fig, axs = fixcorner(fig, axs, ndim)
        figname = 'combo_{}_ssps.pdf'.format(nssps)
        plt.savefig(figname)
        plt.close()
    return

def fixcorner(fig, axs, ndim):
    labelfmt = r'$\rm{{{}}}$'
    for ax in axs.ravel()[:-1*ndim]:
        ax.tick_params(labelbottom='off', tickdir='in')
        ax.axes.set_xlabel('')
    [ax.axes.set_ylabel('') for ax in axs[:, 0]]
    # dmod hack:
    [ax.locator_params(axis='x', nbins=4) for ax in axs.ravel()]
    [ax.locator_params(axis='x', nbins=3) for ax in axs.T[2]]
    fig.text(0.02, 0.5, labelfmt.format('Probability'), ha='center',
             va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0.08, wspace=0.1, right=0.95, top=0.98,
                        bottom=0.08)

    unify_axlims(axs)
    return fig, axs


def unify_axlims(axs, bycolumn=True, x=True, y=False):
    if bycolumn:
        axs = axs.T
    for i in range(len(axs)):
        col = axs[i]
        if x:
            l, h = zip(*[a.get_xlim() for a in col])
            [a.set_xlim(np.min(l), np.max(h)) for a in col]
        if y:
            l, h = zip(*[a.get_ylim() for a in col])
            [a.set_ylim(np.min(l), np.max(h)) for a in col]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="pdf plots for a odyssey calcsfh run")

    parser.add_argument('-t', '--test', action='store_true',
                        help='ssp test')

    parser.add_argument('-o', '--byov', action='store_true',
                        help='filter by core overshoot')
    parser.add_argument('--oned', action='store_true',
                        help='1d pdf plots')

    parser.add_argument('--twod', action='store_true',
                        help='corner plots')

    parser.add_argument('--mist', action='store_true',
                        help='use hard coded mist columns')

    parser.add_argument('--onefig', action='store_true',
                        help='with --oned put all csv files on one plot')

    parser.add_argument('filename', type=str, nargs='*', help='csv file(s)')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.test:
        ssp_test(args.filename)
    elif args.byov:
        bycov(args.filename, oned=args.oned, twod=args.twod, onefig=args.onefig)
    else:
        cluster_result_plots(args.filename, oned=args.oned, twod=args.twod,
                             onefig=args.onefig, mist=args.mist)

if __name__ == "__main__":
    sys.exit(main())
