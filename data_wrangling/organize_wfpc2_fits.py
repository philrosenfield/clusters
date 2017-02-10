#!/astro/apps6/anaconda2.0/bin/python
"""
move all fits files in current directory to directory structure based on fits
header:
./[INSTRUME]/[PROPOSID]_[TARGNAME]/

(will make the directories if they do not exist.)
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

from ..utils import fixnames


def create_dirstruct(table, outfile=None, propid='13901'):
    """
    create a script to move fits files to directory structure from columns
    proposal_id SimbadName. If more than one proposal_id for a given
    SimbadName, it takes passed propid.
    Parameters
    ----------
    table : str
        csv adapted from the discovery portal. See tables/wfpc2/readme.md
    outfile : str [organize.sh]
        file to write bash script
    propid : str [13901]
        proposal id for combined datasets
    """
    tab = fixnames(pd.read_csv(table, header=0))
    line = ''
    for name in np.unique(tab.SimbadName):
        if name == "NONAME":
            continue
        inds, = np.nonzero(tab.SimbadName == name)
        pids = np.unique(tab.proposal_id.iloc[inds])
        if len(pids) > 1:
            pid = propid
        else:
            pid, = pids
        newdir = '{}_{}'.format(pid, name)
        mvfiles = np.unique(tab.Dataset.iloc[inds])

        line += 'mkdir {}\n'.format(newdir)
        line += 'mv {} {}\n'.format('* '.join(mvfiles) + '*', newdir)

    if outfile is None:
        print(line)
    else:
        with open(outfile, 'w') as out:
            out.write(line)
    return


def main(argv=None):
    parser = argparse.ArgumentParser(description=create_dirstruct.__doc__)
    parser.add_argument("-o", "--outfile", type=str, default='organize.sh',
                        help="script to write to")

    parser.add_argument("table", type=str,
                        help="table with SimbadName, proposal_id, Dataset")

    args = parser.parse_args(argv)
    create_dirstruct(args.table, outfile=args.outfile)
    return


if __name__ == "__main__":
    sys.exit(main())
