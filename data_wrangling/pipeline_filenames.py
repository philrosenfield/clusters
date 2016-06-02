import argparse
import os
import sys

from astropy.io import fits

def fix_filename(fname, fext='VEGA', clobber=False):
    name = os.path.split(fname)[1]
    if len(name.split('_')) == 4:
        print('{} seems ok'.format(fname))
    else:
        data = fits.getdata(fname)
        fs = [d.replace('_{}'.format(fext), '') for d in data.dtype.names if fext in d]
        filters = '-'.join(fs)
        if len(filters) == 0:
            print('{} not found. {}'.format(fext, data.dtype.names))
        else:
            pref = fname.split('.')[0]
            ext = '.'.join(fname.split('.')[1:])
            nfname = '_'.join([pref, filters, ext])
            cmd = 'mv -i {} {}'.format(fname, nfname)
            if clobber:
                os.system(cmd)
            else:
                print(cmd)
    return


def main(argv):
    parser = argparse.ArgumentParser(description="Add filter names to file name")

    parser.add_argument('-w', '--wreckless', action='store_true',
                        help='do the mv, not just print mv')

    parser.add_argument('-f', '--filterext', type=str, default='VEGA',
                        help='string next to filter in the columnname e.g., F555W_VEGA')

    parser.add_argument('filenames', nargs='*',
                        help='fits file(s) to work on')

    args = parser.parse_args(argv)
    for fname in args.filenames:
        fix_filename(fname, fext=args.filterext, clobber=args.wreckless)

if __name__ == '__main__':
    main(sys.argv[1:])
