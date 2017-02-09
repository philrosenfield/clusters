"""Make a table from a directory list of fits files to (later) compare against one from MAST portal"""
import argparse
import sys
from astropy.io import fits
from fitshelper.footprints import find_footprint

header="""
#@string, string, string, string, string, string, string, string, string, ra, dec, string, string, int, float, float, float, float, float, string, float, string, string, string, string, float, float, float, string
dataproduct_type,obs_collection,instrument_name,project,filters,wavelength_region,target_name,target_classification,obs_id,s_ra,s_dec,proposal_id,proposal_pi,calib_level,t_min,t_max,t_exptime,em_min,em_max,obs_title,t_obs_release,s_region,jpegURL,obsid,objID,distance,group,litidx,Dataset
"""

def foot2s_region(foot):
   return 'POLYGON J2000 {0!s}'.format(list(foot.ravel()))
                               .replace(',', '')
                               .replace('[','')
                               .replace(']','')

def mast_like(fitsfiles, outfile=None):
    """Make a MAST Portal-like table from a list of fitsfiles."""
    outfile = outfile or 'MASTlike.csv'
    fmt = "image,{TELESCOP:s},{INSTRUME:s}/WFC,,{FILTNAM1:s},,{TARGNAME:s},,,{RA_TARG:f},{DEC_TARG:f},{PROPOSID:d},,,{EXPSTART:f},{EXPEND:f},{EXPTIME:f},,,,,{s_region:s},,,,,,,{ROOTNAME:s}\n"
    line = header

    for f in fitsfiles:
       hdu = fits.open(f)
       hdr = dict(hdu[0].header)
       foot = find_footprint(hdu)
       hdr['s_region'] = foot2s_region(foot)
       line += fmt.format(**hdr)

    with open(outfile, 'w') as outf:
       outf.write(line)
    return outfile


def main(argv=None):
    """main function for mast_like"""
    parser = argparse.ArgumentParser(description=mast_like.__doc__)

    parser.add_argument('-o', '--outfile', type=str,
                        help='output table name')

    parser.add_argument('fnames', type=str, nargs='*',
                        help='fits filename(s)')

    args = parser.parse_args(argv)

    mast_like(args.fnames, outfile=args.outfile)


if __name__ == "__main__":
    sys.exit(main())
