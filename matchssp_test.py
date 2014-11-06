import numpy as np
import os
import matplotlib.pylab as plt
from astroML.stats import binned_statistic_2d
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = False

def strconvert(string):
    col_keys = ['Av', 'IMF', 'dmod', 'logAge', 'mh', 'fit', 'bg1', 'COV']
    if not string in col_keys:
        print ('%s not found in %s' % (string, ', '.join(col_keys)))
        return string

    converters = [r'A_V', 'IMF', r'\mu', r'\log\ {\rm Age}', r'{\rm [M/H]}',
                 'fit', 'bg1', r'\Lambda_c']
    return converters[col_keys.index(string)]

class MatchGrid(object):
    def __init__(self, filenames, covs):
        self.covs = covs
        self.load_grid(filenames)

    def load_grid(self, filenames):
        self.bases = []
        self.names = []
        liness = []

        for filename in filenames:
            base, name = os.path.split(filename)
            self.bases.append(base)
            self.names.append(name)
            with open(filename, 'r') as infile:
                lines = infile.readlines()
            liness.append(lines[:-2])

        col_keys = ['Av', 'IMF', 'dmod', 'logAge', 'mh', 'fit', 'bg1', 'COV']
        dtype = [(c, float) for c in col_keys]
        nrows = len(np.concatenate(liness))
        self.data = np.ndarray(shape=(nrows,), dtype=dtype)
        row = 0
        for i, lines in enumerate(liness):
            for j in range(len(lines)):
                datum = np.array(lines[j].strip().split(), dtype=float)
                datum = np.append(datum, self.covs[i])
                self.data[row] = datum
                row += 1

    def pdf_plot(self):
        ycols = ['mh', 'COV', 'Av']
        xcols = ['logAge', 'logAge', 'dmod']
        zcols = ['fit', 'fit', 'fit']
        stat = np.min
        for i in range(len(ycols)):
            ofigname = 'pdf_%s_%s_%s_%s.png' % \
                (xcols[i], ycols[i], zcols[i], stat)

            if not 'COV' in [xcols[i], ycols[i]]:
                for icov in self.covs:
                    # repeat for each COV
                    inds, = np.nonzero(self.data['COV'] == icov)
                    figname = ofigname.replace('.png', '_cov%.1f.png' % icov)
                    N, xedges, yedges, ax = \
                        self.call_binned_statistic_2d(xcols[i], ycols[i],
                                                      zcols[i], stat=stat,
                                                      inds=inds,
                                                      figname=figname)
                    ax.set_title('$%s=%.1f$' % (strconvert('COV'), icov))
                    plt.savefig(figname)
            N, xedges, yedges, ax = \
                self.call_binned_statistic_2d(xcols[i], ycols[i],
                                              zcols[i], stat=stat,
                                              figname=ofigname)
        return

    def call_binned_statistic_2d(self, xcol, ycol, zcol, stat='median',
                                 inds=None, ax=None, figname=None,
                                 makeplot=True):

        if inds is None:
            inds = np.arange(len(self.data[xcol]))
        xbins = np.unique(self.data[xcol][inds])
        ybins = np.unique(self.data[ycol][inds])
        N, xedges, yedges = binned_statistic_2d(self.data[xcol][inds],
                                                self.data[ycol][inds],
                                                self.data[zcol][inds],
                                                stat, bins=[xbins, ybins])
        if not makeplot:
            return N, xedges, yedges, ''
        imshow_kw = {'cmap': plt.cm.Blues_r, 'interpolation': 'nearest'}

        if ax is None:
            fig, ax = plt.subplots()
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(N.T, extent=extent, **imshow_kw)
        ax.set_xlabel(r'$%s$' % strconvert(xcol), fontsize=20)
        ax.set_ylabel(r'$%s$' % strconvert(ycol), fontsize=20)
        ax.tick_params(labelsize=16)
        if figname is not None:
            plt.savefig(figname)

        return N, xedges, yedges, ax

#N, xedges, yedges = binned_statistic_2d(mg.data['logAge'], mg.data['mh'], mg.data['fit'], 'median')
#imshow(N.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=plt.cm.Blues_r, interpolation='nearest')
