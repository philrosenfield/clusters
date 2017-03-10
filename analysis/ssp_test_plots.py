
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


def cluster_result_plots(sspfns, oned=False, twod=False, onefig=False,
                         mist=False, gauss=True, quantile=True, mock=False,
                         ovis5=False):
    """corner plot of a big combine scrn output"""
    labelfmt = r'$\rm{{{}}}$'
    frompost = False
    mstr = ''
    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']

    # This assures the same order on the plots, though they are default in ssp
    marg_cols = ['Av', 'dmod', 'lage', 'logZ', 'ov']
    if mist:
        marg_cols = ['Av', 'dmod', 'lage', 'logZ', 'vvcrit', 'tbin']
    if mock or ovis5:
        marg_cols = ['Av', 'dmod', 'lage', 'logZ']

    line = ' & '.join(marg_cols) + '\n'

    if mock:
        mstr = '_test'
        ovs = [0.3, 0.4, 0.5, 0.6]
        ssps = [SSP(sspfns[0], gyr=True, filterby={'trueov': ov})
                for ov in ovs]
        name = os.path.splitext(sspfns[0])[0]
        sspfns = ['{0:s}_trueov{1!s}.csv'.format(name, ov) for ov in ovs]
        cmap = plt.cm.Reds
        truth = {'IMF': 1.35,
                 'dmod': 18.45,
                 'Av': 0.1,
                 'dav': 0.0,
                 'dlogZ': 0.1,
                 'bf': 0.3,
                 'dmag_min': -1.5,
                 'vstep': 0.15,
                 'vistep': 0.05,
                 'logZ': -0.40,
                 'sfr': 8e-4,
                 'lagei': 9.1673,
                 'lagef': 9.1847}
    else:
        truth = {}
        ssps = []
        cmap = plt.cm.Blues
        for sspfn in sspfns:
            if sspfn.endswith('.csv'):
                if ovis5:
                    ssp = SSP(sspfn, gyr=True, filterby={'ov': 0.50})
                    cmap = plt.cm.Greens
                else:
                    ssp = SSP(sspfn, gyr=True)
            else:
                ssp = SSP()
                ssp.gyr = True
                ssp.load_posterior(sspfn)
                frompost = True
            ssps.append(ssp)

    nssps = len(ssps)
    ndim = len(marg_cols)
    fig = None
    axs = [None] * nssps
    if onefig:
        fig, axs = plt.subplots(nrows=nssps, ncols=ndim,
                                figsize=(ndim * 1.7, nssps))

    for i, ssp in enumerate(ssps):
        ssp.check_grid(skip_cols=avoid_list)
        sspfn = sspfns[i]
        targ = ssp.name.split('_')[0].upper()
        label = labelfmt.format(targ)
        ylabel = labelfmt.format(targ)
        if onefig or mock:
            label = None

        if oned:
            f, raxs = ssp.pdf_plots(marginals=marg_cols, text=label, axs=axs[i],
                                    quantile=True, fig=fig, frompost=frompost,
                                    gauss1D=gauss, truth=truth)
            if 'lagei' in list(truth.keys()):
                j = marg_cols.index('lage')
                agei = truth['lagei']
                agef = truth['lagef']
                if ssp.gyr:
                    agei = 10 ** (agei - 9)
                    agef = 10 ** (agef - 9)
                raxs[j].fill_betweenx(np.linspace(*raxs[j].get_ylim()), agei,
                                      agef, color='darkred', zorder=0)
            if not onefig:
                figname = sspfn.replace('.csv', '_1d.pdf')
                plt.savefig(figname)
                plt.close()
            else:
                if mock:
                    raxs[-1].set_ylabel(r'$\Lambda_c={!s}$'.format(ovs[i]),
                                        color='darkred')
                    targ = '{!s}'.format(ovs[i])
                else:
                    raxs[-1].set_ylabel(ylabel)
                raxs[-1].yaxis.set_label_position("right")

            if not frompost:
                ssp.write_posterior(filename='{}_post.dat'.format(targ))

        if twod:
            ssp.pdf_plots(marginals=marg_cols, twod=True, quantile=True,
                          cmap=cmap)
            figname = sspfn.replace('.csv', '.pdf')
            plt.savefig(figname)
            plt.close()

        gs = [ssp.__getattribute__(k + 'g') for k in marg_cols]
        fmt = r'${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'
        line += targ + '& '
        try:
            g.mean
            line += ' &  '.join(['{:.3f} & {:.3f}'.format(g.mean/1., g.stddev/2.) for g in gs])
        except:
            try:
                g[2]
            except:
                gs = [ssp.__getattribute__('{0:s}q'.format(q)) for q in marg_cols]
            j = marg_cols.index('logZ')
            print(targ, gs[j])
            gs[j] = 0.01524 * 10 ** gs[j]
            # gs[j][0], gs[j][1] = gs[j][1], gs[j][0]
            line += ' &  '.join([fmt.format(g[2], g[1]-g[2], g[2]-g[0]) for g in gs])

        line += r'\\'
        line += '\n'

    print(line)

    if onefig:
        fig, axs = fixcorner(fig, axs, ndim)
        figname = 'combo_{}_ssp{}s.pdf'.format(nssps, mstr)
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
    fig.text(0.02, 0.5, labelfmt.format('\ln\ Probability'), ha='center',
             va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0.15, wspace=0.15, right=0.95, top=0.98,
                        bottom=0.15)

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

    parser.add_argument('--ov5', action='store_true',
                        help='marginalize over ov=0.5')

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

    cluster_result_plots(args.filename, oned=args.oned, twod=args.twod,
                         onefig=args.onefig, mist=args.mist, mock=args.test,
                         ovis5=args.ov5)

if __name__ == "__main__":
    sys.exit(main())
