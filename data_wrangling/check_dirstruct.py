import os
import sys
from astropy.io import fits
import numpy as np

fitslist = sys.argv[1:]

for f in fitslist:
    inst, pidtarg, fname = f.split('/')
    pid, targ = pidtarg.split('_')
    hdr = fits.getheader(f)

    # does prop id and target in the directory match the header
    if hdr['PROPOSID'] != int(pid):
        print('{}: {} does not match {}'.format(f, hdr['PROPOSID'], int(pid)))
    if hdr['targname'] != targ:
        print('{}: {} does not match {}'.format(f, hdr['targname'], targ))

    # is this the correct version of calwf3
    if f.endswith('flc'):
        cal = float(hdr['CAL_VER'][:3])
        if cal < 3.3:
            print('{}: CAL_VER is less than 3.3: {}'.format(f, hdr['CAL_VER']))

    # if there is no flc file, is there a raw file
    if f.endswith('flt'):
        if not os.path.isfile(f.replace('flt','flc')):
            raw = f.replace('flt', 'raw')
            if not os.path.isfile(raw):
                print '{}: no flc or raw'.format(f)

# are there at least two filters in each directory?
dirs = np.unique([os.path.split(f)[0] for f in fitslist])
for dry in dirs: 
    hdrs = [fits.getheader(os.path.join(dry, f)) for f in os.listdir(dry) if f.endswith('.fits')]
    filts = []
    for h in hdrs:
        try:
            filts.append(h['FILTER'])
        except:
            filts.append(h['FILTER1'])
            filts.append(h['FILTER2'])
    if len(np.unique(filts)) <= 2:
        print('{0} unique filters in {2}: {1}'.format(len(np.unique(filts)), np.unique(filts), dry))


# are targets spread between acs, multiband, wfc3?
os.system('ls -d -1 ACS/** WFC3/** multiband/** > dirlist.dat')
lines = map(str.strip, open('dirlist.dat').readlines())
acs = [l for l in lines if 'ACS' in l]
wfc = [l for l in lines if 'WFC3' in l]
mul = [l for l in lines if 'multiband' in l]
atarg = [m.split('_')[-1] for m in acs]
wtarg = [m.split('_')[-1] for m in wfc]
mtarg = [m.split('_')[-1] for m in mul]
def uniq(arr):
    return (len(np.unique(arr)) - len(arr))

def both(a, b):
    return list(set(a) & set(b))

dets = ['acs', 'wfc', 'multi']
for i, a in enumerate([acs, wfc, mul]):
    print('non-uniques {}: {}'.format(dets[i], uniq(a)))
print('acs - wfc3 same targets: {}'.format(both(atarg, wtarg)))
print('acs - multi same targets: {}'.format(both(atarg, mtarg)))
print('wfc3 - multi same targets: {}'.format(both(wtarg, mtarg)))

