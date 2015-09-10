import argparse
import palettable
from astropy.io import fits
import ResolvedStellarPops as rsp
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import sys

def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None,
              scatter=False):
    '''plot a cmd with errors'''
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
    if not scatter:
        ax.plot(color[inds], mag[inds], '.', color='black', ms=3)
        if not None in [color_err, mag_err]:
            ax.errorbar(color[inds], mag[inds], fmt=None, xerr=color_err[inds],
                        yerr=mag_err[inds], capsize=0, ecolor='gray')
    else:
        if not None in [color_err, mag_err]:
            color_err = color_err[inds]
            mag_err = mag_err[inds]
        ax = rsp.graphics.plotting.scatter_contour(color[inds], mag[inds],
                                              levels=5, bins=200, threshold=400,
                                              log_counts=False, histogram2d_args={},
                                              plot_args={'edgecolors': 'none', 'color': 'k',
                                                         'marker': 'o', 's': 3},
                                              contour_args={},
                                              ax=ax, xerr=color_err,
                                              yerr=mag_err)
    return ax


def _plot_xy(x, y, radii=[100, 500, 1000]):
    hx, bx = np.histogram(x, bins=500)
    hy, by = np.histogram(y, bins=500)
    ix = np.argmax(hx)
    iy = np.argmax(hy)


def add_inset(ax0, extent, xlims, ylims):
    '''add an inset axes to the plot and a rectangle on the main plot'''
    ax = plt.axes(extent)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims[::-1])
    rect = matplotlib.patches.Rectangle((xlims[0], ylims[0]),
                                        np.diff(xlims), np.diff(ylims),
                                        fill=False, color='k')
    ax0.add_patch(rect)
    return ax


def cmd(filename, filter1, filter2, inset=False, scatter=False,
        xyfile=None, fextra='VEGA'):
    '''
    plot cmd of data, two insets are hard coded.
    '''
    gal = rsp.StarPop()
    if xyfile is not None:
        _, _, x, y = np.loadtxt(xyfile, unpack=True)
    else:
        x = np.array([])
        y = np.array([])

    try:
        gal.data = fits.getdata(filename)
        mag = gal.data['{}_{}'.format(filter2, fextra)]
        mag1 = gal.data['{}_{}'.format(filter1, fextra)]
        color = mag1 - mag
        mag_err = gal.data['%s_ERR' % filter2]
        color_err = np.sqrt(gal.data['%s_ERR' % filter1] ** 2 + mag_err ** 2)
        x = gal.data.X
        y = gal.data.Y
    except:
        mag1, mag2 = np.genfromtxt(filename, unpack=True)
        color = mag1 - mag2
        mag = mag2
        mag_err = None
        color_err = None

    good, = np.nonzero((np.abs(color) < 30) & (np.abs(mag) < 30))
    if len(x) == 0:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, (ax, axxy) = plt.subplots(ncols=2, figsize=(16, 8))

    ax = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err, inds=good,
                   ax=ax, scatter=scatter)

    plt.tick_params(labelsize=18)
    #ax.set_ylabel(r'$%s\ %s$' % (filt2, fextra), fontsize=24)
    #ax.set_xlabel(r'$%s-%s\ %s$' % (filt1, filt2, fextra), fontsize=24)
    ax.set_ylabel(r'$%s$' % filter2, fontsize=24)
    ax.set_xlabel(r'$%s-%s$' % (filter1, filter2), fontsize=24)
    ax.set_ylim(26., 14)
    ax.set_xlim(-0.5, 4)

    axs = [ax]

    if inset:
        ax1 = add_inset(ax, [0.45, 0.45, 0.42, 0.3], [0., 0.6], [19.6, 21.7])
        ax1 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax1)
        ax2 = add_inset(ax, [0.18, 0.74, .19, .15], [0.85, 1.1], [18.2, 19.2])
        ax2 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax2)
        axs.extend([ax1, ax2])

    if len(x) > 0:
        axxy.plot(x[good], y[good],  '.', color='black', ms=3)
        axxy.set_ylabel(r'$Y$', fontsize=24)
        axxy.set_xlabel(r'$X$', fontsize=24)
        axs.append(axxy)

    return fig, axs


def overplot_iso(data):
    data = np.genfromtxt('/Users/phil/Downloads/output113116546142.dat')
    fig, axs = data_plots.cmd('../10396_NGC419-HRC.gst.fits', 'F555W_VEGA', 'F814W_VEGA', True)
    mag2 = rsp.astronomy_utils.Mag2mag(data.T[21], 'F814W', 'acs_hrc', dmod=dmod, Av=av)
    mag1 = rsp.astronomy_utils.Mag2mag(data.T[15], 'F555W', 'acs_hrc', dmod=dmod, Av=av)
    icolor = mag1-mag2
    [ax.plot(icolor, mag2, '.', alpha=0.5, color='blue') for ax in axs]


def unique_inds(arr):
    '''return unique values and array of indicies matching the unique value'''
    un_arr = np.unique(arr)
    iarr = np.digitize(arr, bins=un_arr) - 1
    return un_arr, iarr


def plot_isochrone_grid(iso_files, ax_by='age'):
    isos = [rsp.fileio.readfile(i, col_key_line=1) for i in iso_files]
    fnames = [i.split('/')[-1].replace('.dat', '') for i in iso_files]

    ovs = np.array([i.split('_')[2].replace('OV','') for i in fnames],
                   dtype=float)
    un_ovs, iovs = unique_inds(ovs)
    ovstr = r'$\Lambda_c=%.1f$'

    ages = np.array([i.split('_')[3].replace('age','') for i in fnames],
                     dtype=float)
    un_ages, iages = unique_inds(ages)
    agestr = r'$\log Age=%.1f$'

    if ax_by == 'age':
        nax = len(un_ages)
        iax = iages
        icols = iovs
        labs = un_ovs
        labfmt = ovstr
        anns = un_ages
        annfmt = agestr
        ileg = -1
    else:
        nax = len(un_ovs)
        iax = iovs
        icols = iages
        labs = un_ages
        labfmt = agestr
        anns = un_ovs
        annfmt = ovstr
        ileg = 0

    fig, axs = plt.subplots(ncols=nax, figsize=(20, 6), sharex=True,
                            sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.08)

    #colors = brewer2mpl.get_map('RdYlBu', 'Diverging', 5).mpl_colors
    colors = rsp.graphics.discrete_colors(len(labs))
    #colors = ['red', 'black', 'blue', 'orange', 'green']

    # plot the isochrones, each panel at one age, colored by cov
    for i, iso in enumerate(isos):
        axs[iax[i]].plot(iso['logTe'], iso['logLLo'], color=colors[icols[i]],
                           alpha=0.5)

    # fake the legend
    [axs[ileg].plot(-99, -99, color=colors[i], lw=3, alpha=0.3,
                  label=labfmt % labs[i]) for i in range(len(labs))]
    axs[ileg].legend(loc=2, fontsize=16)

    for i, ax in enumerate(axs):
        ax.set_xlabel(r'$\log T_{\rm eff}\ (K)$', fontsize=20)
        ax.grid(color='k')
        ax.annotate(annfmt % anns[i], (3.80, 0.7), fontsize=20)
    axs[0].set_ylim(0.6, 2.2)
    axs[0].set_xlim(4.02, 3.65)
    axs[0].set_ylabel(r'$\log L\ (L_\odot)$', fontsize=20)


def main(argv):
    parser = argparse.ArgumentParser(description="Plot CMD")

    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='output image to write to')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I to plot on yaxis')

    parser.add_argument('-x', '--xyfile', type=str, default=None,
                        help='xyfile to plot')

    parser.add_argument('-f', '--filters', type=str, default=None,
                        help='comma separated filter list.')

    parser.add_argument('-s', '--scatter', action='store_true',
                        help='make a scatter contour plot')

    parser.add_argument('observation', type=str, nargs='*',
                        help='data file to make CMD')


    args = parser.parse_args(argv)
    for obs in args.observation:
        if args.filters is not None:
            filter1, filter2 = args.filters.split(',')
        else:
            target, (filter1, filter2) = rsp.asts.parse_pipeline(obs)

        outfile = args.outfile or obs + '.png'

        fig, axs = cmd(obs, filter1, filter2, inset=False,
                       scatter=args.scatter, xyfile=args.xyfile)

        plt.savefig(outfile)
        print 'wrote {}'.format(outfile)
        plt.close()

if __name__ == '__main__':
    main(sys.argv[1:])
