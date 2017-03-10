import argparse
import sys
import os

import numpy as np
import pandas as pd

from collections import OrderedDict
from scipy import optimize

from match.scripts.utils import fitgauss1D, quantiles


class Posterior(object):
    """Class to access posterior distributions created by SSP.write_posterior"""
    def __init__(self, posterior_file=None):
        if posterior_file is not None:
            self.data = pd.read_csv(posterior_file)
            self.keys = [k for k in self.data.columns if not 'prob' in k]
            self.base, self.name = os.path.split(posterior_file)
            self.dropna_()

    def dropna_(self):
        d = {k: self.data[k].loc[np.isfinite(self.data[k])]
             for k in self.keys}
        self.data_dict = d
        return d

    def best_fit(self):
        assert hasattr(self, 'data'), 'Need to initialize Posterior with file'
        bft = OrderedDict()
        for k in self.keys:
            if not k in bft.keys():
                bft[k] = []
            bft[k].append(self.data[k][np.nanargmax(self.data[k+'prob'])])
        return pd.DataFrame(bft)

    def fitgauss1D(self, attr, norm=False):
        """Fit a 1D Gaussian to a marginalized probability
        see fitgauss1D
        sets attribute 'xattr'g
        """
        x, p = self.safe_select(attr)
        if norm:
            p /= p.max()
        g = utils.fitgauss1D(x, p)
        self.__setattr__('{0:s}g'.format(attr), g)
        return g

    def safe_select(self, attr, maskval=0):
        v = self.data_dict[attr]
        vprob = self.data[attr+'prob'].iloc[v.index]
        vprob[~np.isfinite(vprob)] = maskval
        return v, vprob

    def interpolate_(self, attr, res=200, k=1):
        from scipy.interpolate import splprep, splev
        iatr = 'i{0:s}'.format(attr)
        iatrp = 'i{0:s}prob'.format(attr)
        if not hasattr(self, 'intp_dict'):
            self.intp_dict = {}

        if iatr in self.intp_dict and iatrp in self.intp_dict:
            return self.intp_dict[iatr], self.intp_dict[iatrp]


        x, p = self.safe_select(attr)
        ((tckp, u), fp, ier, msg) = splprep([x, p], k=k, full_output=1)
        ix, ip = splev(np.linspace(0, 1, res), tckp)
        self.intp_dict[iatr] = ix
        self.intp_dict[iatrp] = ip
        return ix, ip

    def correlation(self, attrs='all'):
        import itertools
        from scipy.stats import spearmanr
        _ = [self.interpolate_(a) for a in self.keys]
        if attrs == 'all':
            attrs = [k for k in self.intp_dict.keys()
                     if k.startswith('i') and 'prob' in k]
        else:
            attrs = np.atleast1D(attrs)
            attrs = ['i'+k for k in attrs if not k.startswith('i') or k]
        d = {}
        for xatr, yatr in itertools.product(attrs, attrs):
            if xatr == yatr:
                continue
            rho = spearmanr(self.intp_dict[xatr], self.intp_dict[yatr])
            d['{}_{}'.format(xatr[1:], yatr[1:])] = rho
        return d

def combine_posterior(posts=None, filenames=None, attr='ov'):
    fmt = r'${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'
    if posts is None:
        filenames = np.atleast_1d(filenames)
        posts = [Posterior(f) for f in filenames]
    ovs, ovprobs = zip(*[p.safe_select('ov') for p in posts])

    ovs = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6])
    ovprobs = [ovp.values[:len(ovs)] for ovp in ovprobs]
    ovprobs = np.concatenate(ovprobs).reshape(len(ovs), len(posts))
    # add ln P to multiply posteriors
    sovprob = np.sum(ovprobs.T, axis=1)
    gs = quantiles(ovs, sovprob, maxp=True)
    print(fmt.format(gs[2], gs[1]-gs[2], gs[2]-gs[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="combine PDFs")
    parser.add_argument('filenames', type=str, nargs='*', help='posterior files')
    args = parser.parse_args(argv)
    combine_posterior(filenames=args.filenames)

if __name__ == "__main__":
    sys.exit(main())
