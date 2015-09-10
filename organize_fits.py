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

def main(argv):
    parser = argparse.ArgumentParser(description="Move fits files to directory structure based on fits header INSTRUME/PROPOSID_TARGNAME")
    parser.add_argument("--dryrun", help="do not actually move files or make dirs",
                        action="store_true")

    args = parser.parse_args(argv)
    create_dirstruct(dryrun=args.dryrun)

def create_dirstruct(dryrun=True):
    fitslist = glob.glob1(os.getcwd(), '*fits')
    if len(fitslist) == 0:
        print 'nothing to do.'
    else:
        for fit in fitslist:
            try:
                hdu = fits.open(fit)
            except:
                print('problem with %s' % fit)
                pass
            header = hdu[0].header
            try:
   	       newdir = '%i_%s' % (hdu[0].header['PROPOSID'], hdu[0].header['TARGNAME'])
               newdir = os.path.join(header['INSTRUME'], newdir)
	    except:
               print('error in header. skipping {}'.format(fit))
               continue
            cmd = 'mv -i %s %s' % (fit, newdir)
            if not os.path.isdir(newdir):
                if not dryrun:
                    os.makedirs(newdir)
                else:
                    print 'mkdirs %s' % newdir
            if not dryrun:
	        os.system(cmd)
            else:
                print cmd

if __name__ == "__main__":
    main(sys.argv[1:])
