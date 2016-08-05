from __future__ import print_function
import os
import sys
import argparse
import matplotlib
import ResolvedStellarPops as rsp
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from bokeh.plotting import ColumnDataSource, figure, gridplot
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import Range1d


def bplot_cmd_xy(obs, filter1, filter2, xyfile=None):
    color, mag, _, _, good, x, y = load_obs(obs, filter1, filter2,
                                            xyfile=xyfile)

    pid, target = os.path.split(obs)[1].split('_')[:2]
    title = '{} {}'.format(pid, target)
    xlabel = '{}-{}'.format(filter1, filter2)
    ylabel = '{}'.format(filter2)
    outfile = os.path.split(obs)[1] + '.html'

    data = np.column_stack((color[good], mag[good], x[good], y[good]))

    df = pd.DataFrame(data, columns=['color', 'mag2', 'x', 'y'])
    source = ColumnDataSource(df)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"
    plot_config = dict(plot_width=500, plot_height=500, tools=tools)
    cs = ['#8C3B49', '#212053']

    p1 = figure(title=title, **plot_config)
    p1.circle("color", "mag2", size=1, source=source, color=cs[0])
    p1.y_range = Range1d(26, 14)
    p1.yaxis.axis_label = ylabel
    p1.xaxis.axis_label = xlabel

    p2 = figure(**plot_config)
    p2.circle(data[:, 2], data[:, 3], size=1, source=source, color=cs[1])
    p2.yaxis.axis_label = "Y"
    p2.xaxis.axis_label = "X"

    p = gridplot([[p1, p2]])

    html = file_html(p, CDN, title)

    with open(outfile, 'w') as f:
        f.write(html)
    print('wrote {}'.format(outfile))
    # script, div = components(p)


def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    for old, new in dic.iteritems():
        text = text.replace(old, new)
    return text


def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None,
              scatter=False):
    '''plot a cmd with errors'''
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))
    if not scatter:
        ax.plot(color[inds], mag[inds], '.', color='black', ms=3)
        if None not in [color_err, mag_err]:
            ax.errorbar(color[inds], mag[inds], fmt='none',
                        xerr=color_err[inds], yerr=mag_err[inds],
                        capsize=0, ecolor='gray')
    else:
        if None not in [color_err, mag_err]:
            color_err = color_err[inds]
            mag_err = mag_err[inds]
        # ax = rsp.graphics.plotting.scatter_contour(color[inds], mag[inds], \
        #         levels=5, bins=200, threshold=400, log_counts=False,
        #         plot_args={'edgecolors': 'none', 'color': 'k',
        #                    'marker': 'o', 's': 3}, ax=ax, xerr=color_err,
        #         yerr=mag_err)
    return ax


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


def load_obs(filename, filter1, filter2, xyfile=None, fextra='VEGA'):
    if xyfile is not None:
        _, _, x, y = np.loadtxt(xyfile, unpack=True)
    else:
        x = np.array([])
        y = np.array([])

    if filename.endswith('fits'):
        try:
            data = fits.getdata(filename)
            keyfmt = '{}_{}'
            errfmt = '{}_ERR'
            mag = gal.data[keyfmt.format(filter2, fextra)]
            mag1 = gal.data[keyfmt.format(filter1, fextra)]
            color = mag1 - mag
            mag_err = gal.data[errfmt.format(filter2)]
            color_err = \
                np.sqrt(gal.data[errfmt.format(filter1)] ** 2 + mag_err ** 2)
            x = gal.data.X
            y = gal.data.Y
        except ValueError, err:
            print('Problem with {}: {}'.format(filename, err))
            return None, None
    else:
        mag1, mag2 = np.genfromtxt(filename, unpack=True)
        color = mag1 - mag2
        mag = mag2
        mag_err = None
        color_err = None

    good, = np.nonzero((np.abs(color) < 30) & (np.abs(mag) < 30))
    return color, mag, color_err, mag_err, good, x, y


def cmd(obs, filter1, filter2, inset=False, scatter=False,
        xyfile=None):
    '''
    plot cmd of data, two insets are hard coded.
    '''
    color, mag, color_err, mag_err, good, x, y = load_obs(obs,
                                                          filter1,
                                                          filter2,
                                                          xyfile=xyfile)

    if len(x) == 0:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, (ax, axxy) = plt.subplots(ncols=2, figsize=(16, 8))

    ax = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err, inds=good,
                   ax=ax, scatter=scatter)

    plt.tick_params(labelsize=18)
    ax.set_ylabel(r'${}$'.format(filter2), fontsize=24)
    ax.set_xlabel(r'${}-{}$'.format(filter1, filter2), fontsize=24)

    if filter1 == "F160W" or filter1 == "F110W":
        ax.set_ylim(28., 14)
        ax.set_xlim(-5, 2)
    else:
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
        axxy.plot(x[good], y[good], '.', color='black', ms=3)
        axxy.set_ylabel(r'$Y$', fontsize=24)
        axxy.set_xlabel(r'$X$', fontsize=24)
        axs.append(axxy)

    return fig, axs

# def overplot_iso(data):
#     data = np.genfromtxt('/Users/phil/Downloads/output113116546142.dat')
#     fig, axs = data_plots.cmd('../10396_NGC419-HRC.gst.fits', 'F555W_VEGA',
#                               'F814W_VEGA', True)
#     mag2 = rsp.astronomy_utils.Mag2mag(data.T[21], 'F814W', 'acs_hrc',
#                                         dmod=dmod, Av=av)
#     mag1 = rsp.astronomy_utils.Mag2mag(data.T[15], 'F555W', 'acs_hrc',
#                                          dmod=dmod, Av=av)
#     icolor = mag1-mag2
#     [ax.plot(icolor, mag2, '.', alpha=0.5, color='blue') for ax in axs]


def unique_inds(arr):
    '''return unique values and array of indicies matching the unique value'''
    un_arr = np.unique(arr)
    iarr = np.digitize(arr, bins=un_arr) - 1
    return un_arr, iarr


def plot_isochrone_grid(iso_files, ax_by='age'):
    isos = [fileio.readfile(i, col_key_line=1) for i in iso_files]
    fnames = [i.split('/')[-1].replace('.dat', '') for i in iso_files]

    ovs = np.array([i.split('_')[2].replace('OV', '') for i in fnames],
                   dtype=float)
    un_ovs, iovs = unique_inds(ovs)
    ovstr = r'$\Lambda_c=%.1f$'

    ages = np.array([i.split('_')[3].replace('age', '') for i in fnames],
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

    # colors = brewer2mpl.get_map('RdYlBu', 'Diverging', 5).mpl_colors
    colors = rsp.graphics.discrete_colors(len(labs))
    # colors = ['red', 'black', 'blue', 'orange', 'green']

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


def get_filters(fitsfile):
    h = fits.getheader(fitsfile)
    return [h for h in h['filters'].split(',') if len(h) > 0]


def main(argv):
    parser = argparse.ArgumentParser(description="Plot CMD")

    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='output image to write to ([obs].png)')

    parser.add_argument('-b', '--bokeh', action='store_true',
                        help='make bokeh tables')

    parser.add_argument('-p', '--png', action='store_true',
                        help='make pngs')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I to plot on yaxis (I)')

    parser.add_argument('-x', '--xyfile', type=str, default=None,
                        help='xyfile to plot (read in obs)')

    parser.add_argument('-f', '--filters', type=str, default=None,
                        help='comma separated filter list. (from filename)')

    parser.add_argument('-s', '--scatter', action='store_true',
                        help='make a scatter contour plot')

    parser.add_argument('-c', '--clobber', action='store_true',
                        help='overwrite outfile if exists')

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='invoke pdb')

    parser.add_argument('obs', type=str, nargs='*',
                        help='data file to make CMD')

    args = parser.parse_args(argv)
    if args.pdb:
        import pdb
        pdb.set_trace()

    for obs in args.obs:
        if args.filters is not None:
            filters = args.filters.split(',')
        else:
            _, filters = rsp.asts.parse_pipeline(obs)
        if len(filters) == 1:
            if obs.endswith('.fits'):
                filters = get_filters(obs)
            else:
                print('Error only one filter {}.'.format(obs))
                continue

        filter2 = filters.pop(-1)

        for filter1 in filters:
            # either a supplied outputfile, the obs name + .png
            # or if there are more than 2 filters, the two plotted .png
            outfile = args.outfile
            if outfile is None:
                outfile = obs + '.png'
                if len(filters) > 1:
                    # take out the filters not being plotted
                    try:
                        notfs = [f for f in filters if filter1 not in f]
                        outfile = replace_all(obs + '.png',
                                              dict(zip(notfs,
                                                       ['']*len(notfs))))
                        # ^ leaves _-F336W-F814W or F110W----F814W so:
                        uch = {'--': '-'}
                        outfile = replace_all(replace_all(outfile, uch),
                                              uch).replace('_-', '_')
                    except ValueError, err:
                        print('{}: {}'.format(err, filters))
                        return

            if args.bokeh:
                bplot_cmd_xy(obs, filter1, filter2)

            if args.png:
                if os.path.isfile(outfile) and not args.clobber:
                    print('not overwriting {}'.format(outfile))
                    continue

                _, axs = cmd(obs, filter1, filter2, inset=False,
                             scatter=args.scatter, xyfile=args.xyfile)
                if axs is not None:
                    plt.savefig(outfile)
                    print('wrote {}'.format(outfile))
                    plt.close()

if __name__ == '__main__':
    main(sys.argv[1:])
