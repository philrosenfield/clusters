#!/astro/apps6/anaconda2.0/bin/python
"""
move all fits files in current directory to directory structure based on fits
header:
./[INSTRUME]/[PROPOSID]_[TARGNAME]/

(will make the directories if they do not exist.)
"""
import argparse
import glob
import os
import sys

from astropy.io import fits
import numpy as np
from scipy.spatial import KDTree
from time import localtime, strftime

def unique2d(a):
    return np.array(list(set(tuple(i) for i in a.tolist())))


def separate(tablename, radius=0.1):
    table = pd.read_csv(tablename, header=1)
    k = KDTree(np.column_stack([table['ra'], table['dec']]))
    results = k.query_ball_tree(k, radius)
    isolo = np.concatenate([i for i in results if len(i) == 1])
    nsolo = len(isolo)
    inds, groups = zip(*[(i, r) for (i, r) in
                         enumerate(results) if len(r) > 1])
    groups, igs = np.unique(groups, return_index=True)
    inds = np.array(inds)[igs]
    ngroups = len(groups)
    table['group'] = np.nan
    for i, group in enumerate(groups):
        table.loc[group, 'group'] = i

    table['galaxy'] = np.nan
    table.loc[table['ra'] > 50., 'galaxy'] = 'lmc'
    table.loc[table['ra'] < 50., 'galaxy'] = 'smc'

    # Fix names -- check different targets in each group
    # Rename ANY and SFH..


def nearest_targets(cluster_table, field_table):
    cradec = np.genfromtxt(cluster_table, usecols=(4, 5))
    fradec = np.genfromtxt(field_table, usecols=(4, 5))
    ucradec = unique2d(cradec)
    ufradec = unique2d(fradec)
    k = KDTree(ucradec)
    dists, inds = k.query(ufradec)
    for j, (d, i) in enumerate(zip(dists, inds)):
        ax.plot(uradec[i, 0], uradec[i, 1], 'o')
        ax.plot([uradec[i, 0], ulradec[j, 0]], [uradec[i, 1], ulradec[j, 1]])


def commonnames():
    # SCRAP!
    # grab the table
    # look at the target name
    # check it against simbad ra/dec
    # suggest new target name
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    for i in uinds:
        tt = None
        ct = None
        raw_target = cmast['target'][i]
        if 'NGC' in raw_target:
            targ = raw_target.replace('C-', 'C').split('-')[0]
        else:
            targ = raw_target.replace('-FIELD', '').replace('-COPY', '')
        if targ == 'ANY':
            continue
        c = SkyCoord(ra=cmast['s_ra'][i] * u.degree,
                     dec=cmast['s_dec'][i] * u.degree)
        tt = Simbad.query_object(targ)
        ct = Simbad.query_region(c)
        if tt is None:
            print('name resolver did not find', targ, 'was', raw_target)
            ttname = 'N/D'
        else:
            ttname = tt[0]['MAIN_ID']
        if ct is None:
            print('coord resolver did not find', targ, 'was', raw_target)
            ctname = 'N/D'
        else:
            ctname = ct[0]['MAIN_ID']
            print(targ, raw_target, ttname, ctname)
        for i in range(len(cmast)):
            t = Simbad.query_object(cmast['target'][i].replace('-', ''))
            print(cmast['target'][i], t['MAIN_ID'])


def good_ext(fnames):
    exts = ['raw', 'flt', 'flc', 'c0m', 'c1m']
    f = np.concatenate([[f for f in fnames if ext in f] for ext in exts])
    if len(f) == 0:
        print('No good extensions found {}, {}'.format(fnames, exts))
    return f[0]


def verify(fitsfileslist):
    fitslist = map(str.strip, open(fitsfileslist).readlines())
    for f in fitslist:
        inst, pidtarg, fname = f.split('/')
        pid, targ = pidtarg.split('_')
        hdr = fits.getheader(f)

        if hdr['PROPOSID'] != int(pid):
            print('{}: {} does not match {}'.format(f, hdr['PROPOSID'], pid))
        if hdr['targname'] != targ:
            print('{}: {} does not match {}'.format(f, hdr['targname'], targ))

        if f.endswith('flc'):
            cal = float(hdr['CAL_VER'][:3])
            if cal < 3.3:
                print('{}: CAL_VER is less than 3.3: {}'
                      .format(f, hdr['CAL_VER']))
