import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from match.scripts.cmd import CMD
from match.scripts.fileio import filename_data, get_files
from match.scripts.graphics.graphics import zeroed_cmap, square_aspect
from match.scripts.graphics.graphics import add_inner_title
from match.scripts.ssp import SSP

plt.style.use('presentation')
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)


def comp_cmd(cmd0, cmd, label=None):
    extent = cmd0.extent
    aspect = abs((extent[1] - extent[0]) / (extent[3] - extent[2]))

    hess = cmd0.model - cmd.model
    colors = zeroed_cmap(hess, transparent=True)
    #ind = np.max([np.argmax(colors._lut.T[i])
    #              for i in range(len(colors._lut.T)-1)])
    #colors._lut[ind][-1] = 0
    fig, ax = plt.subplots(figsize=(5, 4.5))
    kw = {'extent': extent, 'origin': 'upper', 'interpolation': 'nearest',
          'aspect': aspect}
    ax.imshow(cmd0.data, **kw)
    im = ax.imshow(hess, cmap=colors, alpha=0.5, **kw)
    if label is not None:
        add_inner_title(ax, label, loc=4)
    cb = plt.colorbar(im)
    cmd0.set_axis_labels(ax=ax)
    return fig, ax


def diff_(cmd0, cmd1, ssp=None):
    d0 = filename_data(cmd0.name, exclude='')
    d1 = filename_data(cmd1.name, exclude='')
    if ssp:
        d0.update(ssp.data.iloc[np.argmin(np.abs(ssp.data.fit - cmd0.fit))].to_dict())
        d1.update(ssp.data.iloc[np.argmin(np.abs(ssp.data.fit - cmd1.fit))].to_dict())
    a = {o : (d1[o], d0[o]) for o in set(d0.keys()).intersection(d1.keys()) if d0[o] != d1[o]}
    return a


def twobest(sstr, ssp, label=None):
    cmds = [CMD(c) for c in get_files(places(ssp.name.split('_')[0]), sstr)]
    icmds = np.argsort([cmd.fit for cmd in cmds])[:2]
    cmd0 = cmds[icmds[0]]
    cmd = cmds[icmds[1]]
    print(diff_(cmd0, cmd, ssp=ssp))
    return comp_cmd(cmd0, cmd, label=label)

def places(target):
    data_base = '/Volumes/tehom/research/clusters/asteca/acs_wfc3/paper1/final_data/'
    places_dict = {'NGC1644': os.path.join(data_base, 'NGC1644', 'lowz', 'slurm'),
                   'NGC2213': os.path.join(data_base, 'NGC2213', 'slurm'),
                   'NGC2203': os.path.join(data_base, 'NGC2203', 'slurm'),
                   'NGC2173': os.path.join(data_base, 'NGC2173', 'slurm'),
                   'NGC1978': os.path.join(data_base, 'NGC1978', 'slurm'),
                   'NGC1917': os.path.join(data_base, 'NGC1917', 'slurm'),
                   'NGC1795': os.path.join(data_base, 'NGC1795', 'slurm'),
                   'HODGE2': os.path.join(data_base, 'HODGE2', 'slurm')}
    return places_dict[target.upper()]

def interesting_plots(outdir=None):
    """CMD plots at edges of parameter space"""
    outdir = outdir or '/Users/rosenfield/Desktop/'

    fig, ax = twobest('*bf?.??_imf0.75_*ov0.5*cmd', ssp1644, label=r'$\rm{NGC1644}$')
    plt.savefig(os.path.join(outdir, 'NGC1644_bfcomp.pdf'))

    fig, ax = twobest('*bf0.75_imf*ov0.50*cmd', ssp2213, label=r'$\rm{NGC2213}$')
    plt.savefig(os.path.join(outdir, 'NGC2213_imfcomp.pdf'))

    fig, ax = twobest('*bf0.65_imf*cmd', ssp2203, label=r'$\rm{NGC2203}$')
    plt.savefig(os.path.join(outdir, 'NGC2203_imfcomp.pdf'))

    fig, ax = twobest('*bf0.65_imf0.5_*cmd', ssp2203, label=r'$\rm{NGC2203}$')
    plt.savefig(os.path.join(outdir, 'NGC2203_ovcomp.pdf'))
    ax.set_xlim(0.4, 1.65)
    ax.set_ylim(22., 19.5)
    square_aspect(ax)
    plt.savefig(os.path.join(outdir, 'NGC2203_ovcomp_zoom.pdf'))

    fig, ax = twobest('*bf0.55_imf*ov0.50*cmd', ssp2173, label=r'$\rm{NGC2173}$')
    plt.savefig(os.path.join(outdir, 'NGC2173_imfcomp.pdf'))

    fig, ax = twobest('*bf0.75_imf*cmd', ssp1978, label=r'$\rm{NGC1978}$')
    plt.savefig(os.path.join(outdir, 'NGC1978_imfcomp.pdf'))

    fig, ax = twobest('*bf0.75_imf0.5_*cmd', ssp1978, label=r'$\rm{NGC1978}$')
    plt.savefig(os.path.join(outdir, 'NGC1978_ovcomp.pdf'))
    ax.set_xlim(0.4, 1.2)
    ax.set_ylim(22., 18.5)
    square_aspect(ax)
    plt.savefig(os.path.join(outdir, 'NGC1978_ovcomp_zoom.pdf'))


def best_cmds(outdir=None):
    # best fit ssp run (found using ipython):
    # ssp.data.ssp.iloc[np.argmin(ssp.data.fit)]
    outdir = outdir or '/Users/rosenfield/Desktop/'

    bestfit = {'NGC1917': 1578,
               'NGC1795': 1641,
               'HODGE2': 1675,
               'NGC1644': 1776,
               'NGC2213': 1216,
               'NGC2203': 1814,
               'NGC2173': 1531,
               'NGC1978': 451}
               # Done forget NGC1718...
    for target in bestfit.keys():
        figname = os.path.join(outdir, '{:s}_cmd.pdf'.format(target))
        cmd = CMD(get_files(places(target),
                            '*ssp{:d}.out.cmd'.format(bestfit[target]))[0])
        labels = cmd.set_labels()
        labels[-1] = r'$\rm{sig}$'
        cmd.pgcmd(labels=labels, figname=figname, twobytwo=False)
    return

best_cmds()
res_base = '/Volumes/tehom/research/clusters/asteca/acs_wfc3/paper1/final_data/results'
ssp1644 = SSP(os.path.join(res_base, 'NGC1644_full.csv'))
ssp2213 = SSP(os.path.join(res_base, 'NGC2213_full.csv'))
ssp2203 = SSP(os.path.join(res_base, 'NGC2203_full.csv'))
ssp2173 = SSP(os.path.join(res_base, 'NGC2173_full.csv'))
ssp1978 = SSP(os.path.join(res_base, 'NGC1978_full.csv'))
ssp1917 = SSP(os.path.join(res_base, 'NGC1917_full.csv'))
ssph2 = SSP(os.path.join(res_base, 'HODGE2_full.csv'))
ssp1795 = SSP(os.path.join(res_base, 'NGC1795_full.csv'))
interesting_plots()
