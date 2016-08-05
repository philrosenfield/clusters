from __future__ import print_function
import os
import numpy as np
from astropy.io import fits
from update_key import revert_key

def fix_multiband(root_path):
    line = 'mkdir ACS; mkdir WFC3\n'
    for root, _, filenames in os.walk(root_path):
        key = 'PROPOSID'
        fitsfiles = [f for f in filenames if f.endswith('.fits')]
        nfits = len(fitsfiles)
        subdir = os.path.split(root)[1]
        cams = []
        opids = []
        for f in fitsfiles:
            hdr = fits.getheader(os.path.join(root, f))
            updates = [i for i in hdr['history'] if key.upper() in i.upper()]
            try:
                opid, = [a for a in [int(u.split('from')[1].split('to')[0])
                                     for u in updates] if a != 13901]
            except ValueError:
                print('{}: {} {} never updated?!'.format(f, hdr[key], hdr['targname']))
                break
            camera = hdr['instrume']
            opids.append(opid)
            cams.append(camera)
        ucams = np.unique(cams)
        uopids = np.unique(opids)
        msg = ''
        cmsg = 'Same'
        pmsg = 'Same'
        if len(ucams) != 1:
            cmsg = 'Multi'
        if len(uopids) != 1:
            pmsg = 'Multi'
    
        msg = '{} fitsfiles in {}. {} cams: {}; {} pids: {}'.format(nfits, subdir,
                                                                    cmsg, ucams,
                                                                    pmsg, uopids)
        print(msg)
    
        if cmsg == pmsg == 'Same':
            curkey, targ = os.path.split(subdir)[1].split('_')
            npid, = uopids
            newdir = '_'.join(['{}'.format(npid), targ])
            fnames = [os.path.join(root, f) for f in fitsfiles]
            revert_key(key, fnames=fnames)
            line += 'mkdir {}/{}\n'.format(ucams[0], newdir)
            line += '\n'.join(['mv {} {}/{}/'.format(f.replace('.fits', '*'),
                                                     ucams[0], newdir)
                               for f in fnames])
            line += '\n'
    
    with open('fixmultiband.sh', 'w') as outp:
        outp.write(line)
    return line

if __name__ == "__main__":
    fix_multiband(os.getcwd())
