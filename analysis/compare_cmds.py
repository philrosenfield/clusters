"""Functions to print the best CMDs and compare them to the next best CMD"""
import os
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from match.scripts.cmd import CMD, sortbyfit
from match.scripts.compare_cmds import comp_cmd, diff_
from match.scripts.fileio import get_files
from match.scripts.graphics.graphics import square_aspect
from match.scripts.ssp import SSP
from match.scripts.config import OUTEXT, EXT

CMDEXT = OUTEXT + '.cmd'

plt.style.use('presentation')
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)


def two_best(loc, ssp=None, label=None, figname=None):
    """plot the difference between best two hess diagrams and print differences"""
    all_cmdfns = get_files(loc, '*{}'.format(CMDEXT))
    if len(all_cmdfns) == 0:
        print('no {0:s} files found in {1:s}'.format(CMDEXT, loc))
        return
    twobest_cmdfns = sortbyfit(all_cmdfns, onlyheader=True)[:2]

    cmd0, cmd = [CMD(c) for c in twobest_cmdfns]
    if ssp is None:
        try:
            sspfn, = get_files(loc, '*csv')
            ssp = SSP(sspfn, gyr=True)
            label = r'$\rm{{{}}}$'.format(ssp.name.split('_')[0])
        except:
            label = None

    print(cmd0.name)
    dif = diff_(cmd0, cmd, ssp=ssp)
    print(dif)
    for key in dif.keys():
        print(key, np.diff(np.array(dif[key].split(','), dtype=float))[0])

    comp_cmd(cmd0, cmd, label=label, figname=figname)

    return

def two_bests(targets, outdir=None, data_base=None):
    """call two_best with list of targets"""
    outdir = outdir or os.getcwd()
    places_dict = places(targets, data_base=data_base)

    for target in targets:
        figname = os.path.join(outdir, '{:s}_toptwo.pdf'.format(target))
        two_best(places_dict[target], figname=figname)
    return

def places(targets, data_base=None, subdir='slurm'):
    """load a dictionary of targets and the location of their data"""
    data_bases = data_base or os.getcwd()
    places_dict = {}
    for targ in targets:
        places_dict[targ] = os.path.join(data_base, targ, subdir)
        assert os.path.isdir(places_dict[targ]), \
            'Directory does not exist {}'.format(places_dict[targ])
    return places_dict


def best_pgcmd(loc, figname=None, twobytwo=False, sig=False):
    """call pgcmd for the best fitting cmd file"""
    all_cmdfns = get_files(loc, '*{}'.format(CMDEXT))
    if len(all_cmdfns) == 0:
        print('no {0:s} files found in {1:s}'.format(CMDEXT, loc))
        return
    best_cmdfn = sortbyfit(all_cmdfns, onlyheader=True)[0]
    if figname is None:
        figname = best_cmdfn + EXT

    cmd = CMD(best_cmdfn)
    cmd.pgcmd(figname=figname, twobytwo=twobytwo, sig=sig)
    return


def best_pgcmds(targets, outdir=None, data_base=None):
    """Call best_pgcmd with list of targets"""
    outdir = outdir or os.getcwd()
    places_dict = places(targets, data_base=data_base)

    for target in targets:
        figname = os.path.join(outdir, '{:s}_cmd.pdf'.format(target))
        best_pgcmd(places_dict[target], figname=figname)
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(description="plot and compare cmd files")

    parser.add_argument('targets', type=str,
                        help='file with a list of targets names')

    parser.add_argument('--outdir', type=str, default=os.getcwd(),
                        help='directory to put plots')

    parser.add_argument('--data_base', type=str, default=os.getcwd(),
                        help='where to look for the data in data/cluster/slurm')

    parser.add_argument('-b', '--best', action='store_true',
                        help='plot best fit cmd comparison')

    parser.add_argument('-c', '--comp', action='store_true',
                        help='plot comparison of top two best fits')
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    with open(args.targets) as inp:
        targets = [i.strip() for i in inp.readlines() if len(i.strip()) > 0]

    if args.best:
        best_pgcmds(targets, outdir=args.outdir, data_base=args.data_base)

    if args.comp:
        two_bests(targets, outdir=args.outdir, data_base=args.data_base)
    return

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
