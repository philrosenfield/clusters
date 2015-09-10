#!/astro/apps6/anaconda2.0/bin/python
import argparse
from astropy.io import fits
import sys
import os

def make_table(filelist, outfile='table.dat', clobber=False):
    line = ('# instrument detector propid target ra dec filter1 filter2 pr_inv '
            'exptime date-obs \n')
    for filename in filelist:
        data = fits.getheader(filename)
        try:
            line += ('%(INSTRUME)s %(DETECTOR)s %(PROPOSID)s %(TARGNAME)s '
                     '%(RA_TARG).6f %(DEC_TARG).6f %(FILTER1)s %(FILTER2)s '
                     '%(PR_INV_L)s_%(PR_INV_F)s %(EXPTIME)i %(DATE-OBS)s ' % data)
            line += '%s \n' % filename
        except KeyError, e:
            try:
                line += ('%(INSTRUME)s %(DETECTOR)s %(PROPOSID)s %(TARGNAME)s '
                         '%(RA_TARG).6f %(DEC_TARG).6f ... %(FILTER2)s '
                         '%(PR_INV_L)s_%(PR_INV_F)s %(EXPTIME)i %(DATE-OBS)s ' % data)
                line += '%s \n' % filename
            except KeyError, e:
                try:
                    line += ('%(INSTRUME)s %(DETECTOR)s %(PROPOSID)s %(TARGNAME)s '
                             '%(RA_TARG).6f %(DEC_TARG).6f %(FILTER1)s ... '
                             '%(PR_INV_L)s_%(PR_INV_F)s %(EXPTIME)i %(DATE-OBS)s ' % data)
                    line += '%s \n' % filename
                except KeyError, e:
                    try:
                        line += ('%(INSTRUME)s %(DETECTOR)s %(PROPOSID)s %(TARGNAME)s '
                                 '%(RA_TARG).6f %(DEC_TARG).6f ... ... '
                                 '%(PR_INV_L)s_%(PR_INV_F)s %(EXPTIME)i %(DATE-OBS)s ' % data)
                        line += '%s \n' % filename
                    except KeyError, e:
                        print('{}: {}'.format(filename, e))

    if clobber or not os.path.isfile(outfile):
        wflag = 'w'
    else:
        print('appending to {}'.format(outfile))
        wflag = 'a'
        line = line[1:]

    with open(outfile, wflag) as out:
       out.write(line)


def main(argv):
    parser = argparse.ArgumentParser(description="Create a table from fits file header information\ne.g., make_table.py */ACS/*/*fits")

    parser.add_argument('name', nargs='*', type=str, help='fits files')

    parser.add_argument('-o', '--outfile', type=str, default='table.dat',
                        help='specify output file name')

    parser.add_argument('-f', '--clobber', action='store_true',
                        help='overwite outfile if it already exists, otherwise append')

    args = parser.parse_args(argv)

    make_table(args.name, outfile=args.outfile, clobber=args.clobber)


if __name__ == '__main__':
    main(sys.argv[1:])
