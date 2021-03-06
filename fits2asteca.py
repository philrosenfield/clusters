import argparse
import os
import sys
import numpy as np

from astropy.io import fits
from match.scripts.match_phot import asteca_fmt

def main(argv=None):
    parser = argparse.ArgumentParser(description="fits file to asteca")
    parser.add_argument('fitsnames', type=str, nargs='*', help='fits file(s)')
    args =  parser.parse_args(argv)

    for fitsname in args.fitsnames:
        fname = fitsname.replace('fits', 'asteca')
        data = fits.getdata(fitsname)

        filters = [c for c in data.dtype.names if c.endswith('VEGA')]
        filter2 = 'F814W_VEGA'
        filters.pop(filters.index(filter2))
        for filter1 in filters:
            fnamesp = fname.split('_')
            fnamesp[2] = '-'.join([filter1, filter2])
            nfname = '_'.join(fnamesp).replace('_VEGA', '')
            np.savetxt(nfname, asteca_fmt(data, filter1, filter2, crowd=1.3),
                       fmt='%.6f')
            print('wrote {}'.format(nfname))

if __name__ == "__main__":
    sys.exit(main())
