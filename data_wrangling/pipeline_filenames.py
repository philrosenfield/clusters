import argparse
import os
import sys

from astropy.io import fits

def fix_filename(fname, fext='VEGA', clobber=False, newdir=None, extra=''):
    newdir = newdir or os.getcwd()
    base, name = os.path.split(fname)

    mv = 'mv -i'
    if os.path.isdir(newdir):
        base = newdir
        mv = 'cp'
    if len(name.split('_')) == 4:
        print('{} seems ok'.format(fname))
    else:
        data = fits.getdata(fname)
        fs = [d.replace('_{}'.format(fext), '') for d in data.dtype.names if fext in d]
        filters = '-'.join(fs)
        if len(filters) == 0:
            print('{} not found. {}'.format(fext, data.dtype.names))
        else:
            pref = name.split('.')[0]
            ext = '.'.join(name.split('.')[1:])
            if not ext.startswith('.'):
                ext = '.' + ext
            nname = '_'.join([pref, filters, extra, ext]).replace('__', '')
            nfname = os.path.join(base, nname)
            cmd = '{} {} {}'.format(mv, fname, nfname)
            print(cmd)
            if clobber:
                os.system(cmd)
                
    return


def main(argv):
    parser = argparse.ArgumentParser(description="Add filter names to file name")

    parser.add_argument('-w', '--wreckless', action='store_true',
                        help='do the mv/cp, not just print mv/cp')

    parser.add_argument('-f', '--filterext', type=str, default='VEGA',
                        help='string next to filter in the columnname e.g., F555W_VEGA')

    parser.add_argument('-e', '--extra', type=str, default='',
                        help='extra string to add to filename')

    parser.add_argument('-o', '--outdir', type=str,
                        help='output directory if different')

    parser.add_argument('filenames', nargs='*',
                        help='fits file(s) to work on')

    args = parser.parse_args(argv)
    for fname in args.filenames:
        fix_filename(fname, fext=args.filterext, clobber=args.wreckless,
                     newdir=args.outdir, extra=args.extra)

if __name__ == '__main__':
    main(sys.argv[1:])
