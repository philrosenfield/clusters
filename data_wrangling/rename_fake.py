"""
Write a script to rename Cliff's fake output format to 
PID_TARGET_filters.extensions
"""
import os
import glob

phot = glob.glob('*.match')
fake = glob.glob('*fake.dat')

line = ''
# housing for phot files that do not have ASTs
if not os.path.isdir('unmatched'):
    line += 'mkdir unmatched\n'

nfs = []
# First rename the fake files
for i, f in enumerate(fake):
    newfake = f.replace('.gst', '').replace('.fake.dat', '.gst.matchfake').replace('W_F', 'W-F')
    line += 'mv {} {}\n'.format(f, newfake)
    nfs.append(newfake)

for p in phot:
    f = [a for a in nfs if a.replace('fake', '') in p]
    pid = p.split('_')[0]
    if len(f) > 0:
        # phot and fake file
        line += 'mv {0} {1}_{0}\n'.format(f[0], pid)
    else:
        # only phot file
        line += 'mv {} unmatched/ \n'.format(p)

with open('rename_fake.sh', 'w') as outp:
    outp.write(line)
