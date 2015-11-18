#!/astro/apps6/anaconda2.0/bin/python
"""
make a script to move all fits files in current directory to directory structure
based on its header:
./[INSTRUME]/[PROPOSID]_[TARGNAME]/

(will make the directories if they do not exist.)
"""
import argparse
import glob
import os
import sys

from astropy.io import fits

def main(argv):
    parser = argparse.ArgumentParser(description="Move fits files to directory structure based on fits header INSTRUME/PROPOSID_TARGNAME")
    parser.add_argument("-o", "--outfile", type=str, default='organaize.sh',
                        help="script name to write to")

    args = parser.parse_args(argv)
    create_dirstruct(outfile=args.outfile)

def create_dirstruct(outfile=None):
    line = ''
    fitslist = glob.glob1(os.getcwd(), '*fits')
    if len(fitslist) == 0:
        line += 'nothing to do.\n'
    else:
        for fit in fitslist:
            try:
                hdu = fits.open(fit)
            except:
                line += '# problem with %s\n' % fit)
                pass
            header = hdu[0].header
            try:
                newdir = '%i_%s' % (hdu[0].header['PROPOSID'], hdu[0].header['TARGNAME'])
                newdir = os.path.join(header['INSTRUME'], newdir)
            except:
                line += 'error in header. skipping {}\n'.format(fit)
                continue

            if not os.path.isdir(newdir):
               line += 'mkdirs %s\n' % newdir
            cmd = 'mv -i %s %s\n' % (fit, newdir)
            line += cmd

    if outfile is None:
        print(line)
    else:
        with open(outfile, 'r') as out:
            out.write(line)


if __name__ == "__main__":


    main(sys.argv[1:])
