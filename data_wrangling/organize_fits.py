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


def same_targ(fits):
    return np.unique([f.split('_')[0] for f in fits], return_index=True)


def create_dirstruct(outfile=None, data=None, propid=None, basedir=None):
    """
    e.g.,
    #data = pd.read_csv('multiband.csv', header=0)
    #create_dirstruct(outfile='multiband.sh', data=data, propid=13901, basedir='multiband')
    #data = pd.read_csv('final_acs.csv', header=0)
    #create_dirstruct(outfile='acs.sh', data=data)
    #data = pd.read_csv('final_wfc3.csv', header=0)
    #create_dirstruct(outfile='wfc3.sh', data=data)

    """
    line = ''
    if data is None:
        fitslist = glob.glob1(os.getcwd(), '*fits')
        fnames, inds = same_targ(fitslist)
    else:
        fitslist = data['filename']

    if len(fitslist) == 0:
        print('nothing to do.')
        return

    print('{} fits files'.format(len(fitslist)))
    for i in range(len(fitslist)):
        fname = fitslist[i]
        fnames = [fname]
        flc = fname.replace('flt', 'flc')
        raw = fname.replace('flt', 'raw')
        if os.path.isfile(flc):
            fnames.append(flc)
        if os.path.isfile(raw):
            fnames.append(raw)
        for f in fnames:
            try:
                print('reading', f)
                hdu = fits.open(f)
                hdr = hdu[0].header
            except:

                if not os.path.isfile(f):
                    print('{} not found, already moved?'.format(f))
                else:
                    line += 'echo "problem with {}"\n'.format(f)
                continue
            try:
                if propid is None:
                    tpropid = hdr['PROPOSID']
                else:
                    tpropid = propid

                if basedir is None:
                    tbasedir = hdr['INSTRUME']
                else:
                    tbasedir = basedir

                newdir = '%i_%s' % (tpropid, hdr['TARGNAME'])
                newdir = os.path.join(tbasedir, newdir)
            except:
                line += 'echo "error in header. skipping {}"\n'.format(f)
                continue

            line += 'echo "mv -i {} {}:"\n'.format(f, newdir)
            line += 'mv -i {} {}\n'.format(f, newdir)

            if not os.path.isdir(newdir):
                os.makedirs('{}'.format(newdir))
                print('made dir {}'.format(newdir))

    if outfile is None:
        print(line)
    else:
        with open(outfile, 'w') as out:
            out.write(line)


desc = """
Move fits files to directory structure based on fits header
INSTRUME/PROPOSID_TARGNAME
"""
def main(argv=None):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--outfile", type=str, default='organize.sh',
                        help="script to write to")

    args = parser.parse_args(argv)
    create_dirstruct(outfile=args.outfile)
    return


if __name__ == "__main__":
    sys.exit(main())
