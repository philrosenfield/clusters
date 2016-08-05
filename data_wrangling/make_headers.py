import sys
from astropy.io import fits

fitslist = sys.argv[1:]
for f in fitslist:
    h = fits.getheader(f)
    h.tofile(f + '.hdr', clobber=True, sep='\n', padding=False)

