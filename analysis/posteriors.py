import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import optimize
from collections import OrderedDict, defaultdict
import seaborn as sns

plt.style.use('presentation')
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)


def gaussian(p, x):
    """ p = [mu, sig, max=[1.], offset=[0.]]"""
    if np.shape(p)[0] == 2:
        p = np.append(p, 1.)
    if np.shape(p)[0] == 3:
        p = np.append(p, 0.)

    return p[3] + p[2] / (p[1] * np.sqrt(2 * np.pi)) * \
            np.exp(-((x - p[0]) ** 2 / (2 * p[1] **2 )))

def errfunc(p, x, y):
    return y - gaussian(p, x)


class Posterior(object):
    """Class to access posterior distributions created by SSP.write_posterior"""
    def __init__(self, posterior_file=None):
        if posterior_file is not None:
            self.data = pd.read_csv(posterior_file)
            self.keys = [k for k in self.data.columns if not 'prob' in k]
            self.base, self.name = os.path.split(posterior_file)
            self.dropna_()

    def dropna_(self):
        d = {k: post.data[k].loc[np.isfinite(post.data[k])]
             for k in post.data.columns}
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

    def fitgauss1D(self, attr):
        """Fit a 1D Gaussian to a marginalized probability
        see .utils.fitgauss1D
        sets attribute 'xattr'g
        """
        x, p = self.safe_select(attr)
        g = utils.fitgauss1D(x, p)
        self.__setattr__('{0:s}g'.format(attr), g)
        return g

    def safe_select(self, attr):
        return self.data_dict[attr], self.data_dict[attr + 'prob']

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

def best_table(posterior_files):
    df = pd.DataFrame()
    for filename in posterior_files:
        target = filename.split('_')[0]
        post = Posterior(filename)
        bf = post.best_fit()
        bf['target'] = target
        df = df.append(bf, ignore_index=True)
    print(df.to_latex())
    return df

def age_ov(posterior_files, gi09=False):
    df = best_table(posterior_files)
    fig, ax = plt.subplots()
    ax.errorbar(df.lage, df.ov, xerr=0.06, yerr=0.05, fmt='none', elinewidth=3,
                color='k', label='')
    ax.plot(df.lage, df.ov, 'o', ms=10, color='darkred', label=r'$\rm{This\ work}$')
    if gi09:
        # $\lambdac=0.47^{+0.14}_{-0.04}$ and $\log$ Age=$1.35^{+0.11}_{-0.04}$
        ax.errorbar(np.array([1.35]), np.array([0.47]),
                    xerr=[np.array([0.11]), np.array([0.04])],
                    yerr=[np.array([0.14]), np.array([0.04])],
                    fmt='none', elinewidth=3, color='k', label='')
        ax.plot(1.35, 0.47, 'o', ms=10, color='gray', label=r'$\rm{Girardi\ et\ al.\ (2009)}$')

    ax.set_xlabel(r'${\rm Cluster\ Age\ (Gyr)}$')
    ax.set_ylabel(r'$\Lambda_{\rm c}\ (H_p)$')
    plt.legend(loc='best')
    plt.savefig('age_ov.pdf')
    return ax


def calc_uncertainties(posterior_file):
    post = Posterior(posterior_file)
    bf = post.best_fit()
    print(post.name)
    for key in post.keys:
        sig0 = 2 * np.unique(np.diff(post.data[key]))[0]
        p = [bf[key][0], sig0]
        y = post.data[key+'prob'][np.isfinite(post.data[key+'prob'])]
        x = post.data[key][np.isfinite(post.data[key])]
        fit, *rest = optimize.leastsq(errfunc, p, args=(x, y),
                                     full_output=True)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        fwhm = 2 * np.log(2) * fit[1]
        if rest[-1] > 4 or np.abs(fwhm) < sig0 / 2:
            # print(rest[-2])
            fwhm = sig0 / 2
            bester = bf[key][0]
        else:
            print(key, sig0)
            fwhm = 2 * np.log(2) * fit[1]
            xx = np.linspace(x.iloc[0], x.iloc[-1], 100)
            bester = xx[np.argmax(gaussian(fit, xx))]
            ax.plot(xx, gaussian(fit, xx))
            print('{:s} {:f} {:f}'.format(key, bester, fwhm))
