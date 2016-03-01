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

def unique2d(a):
    return np.array(list(set(tuple(i) for i in a.tolist())))

def nearest_targets(cluster_table, field_table):
    cradec = np.genfromtxt(cluster_table, usecols=(4,5))
    fradec = np.genfromtxt(field_table, usecols=(4,5))
    ucradec = unique2d(cradec)
    ufradec = unique2d(fradec)
    k = KDTree(ucradec)
    dists, inds = k.query(ufradec)
    for j, (d, i) in enumerate(zip(dists, inds)):
        ax.plot(uradec[i,0], uradec[i,1], 'o')
        ax.plot([uradec[i,0], ulradec[j, 0]] , [uradec[i,1], ulradec[j, 1]])
    
def good_ext(fnames):
    exts = ['drz', 'flt', 'flc', 'c0m', 'c1m']
    f = np.concatenate([[f for f in fnames if ext in f] for ext in exts])
    return f[0]

def main(argv):
    parser = argparse.ArgumentParser(description="Move fits files to directory structure based on fits header I
NSTRUME/PROPOSID_TARGNAME")
    parser.add_argument("-o", "--outfile", type=str, default='organize.sh',
	                help="script to write to")

    args = parser.parse_args(argv)
    create_dirstruct(outfile=args.outfile)

def same_targ(fits):
    return np.unique([f.split('_')[0] for f in fits], return_index=True)

def create_dirstruct(outfile=None):
    line = ''
    fitslist = glob.glob1(os.getcwd(), '*fits')
    if len(fitslist) == 0:
        line += 'nothing to do.\n'
    else:
        import pdb; pdb.set_trace()
	fnames, inds = same_targ(fitslist)
	print('{} fits files'.format(len(fitslist)))
        for i in range(len(fnames)):
            if i+1 % 100 == 0:
                print('{} of {}'.format(i+1, len(fnames)))
            try:
                fit = good_ext(fitslist[inds[i]:inds[i+1]])
		hdu = fits.open(fit)
            except:
                line += '# problem with %s\n' % fit
                pass
            header = hdu[0].header
            try:
   	       newdir = '%i_%s' % (hdu[0].header['PROPOSID'], hdu[0].header['TARGNAME'])
               newdir = os.path.join(header['INSTRUME'], newdir)
	    except:
               line += 'error in header. skipping {}\n'.format(fit)
               continue
            cmd = 'mv -i %s*fits %s\n' % (fnames[i], newdir)
            if not os.path.isdir(newdir):
                os.makedirs('{}'.format(newdir))
                print('made dir {}'.format(newdir))
            line += cmd

    if outfile is None:
        print(line)
    else:
	with open(outfile, 'w') as out:
            out.write(line)


if __name__ == "__main__":
    main(sys.argv[1:])
