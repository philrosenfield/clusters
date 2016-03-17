import numpy as np
import sys

#generate match parameter files

# might be constant values
# dmod
dmod0 = 18.5
dmod1 = 18.5
dmod_delta = 0.1
# Av
av0 = 0.0
av = 1.0
av_delta = 0.1
# LogZ
zmin = -1.6
zmax = -0.6
z_delta = 0.10
# CMD info
mbin = 0.1
cbin = 0.05
cmin = -0.5
cmax = 1.6
filter1 = 'F555W'
filter2 = 'F814W'
f1_brightlim = 13.
f1_faintlim = np.float(sys.argv[1])
f2_brightlim = 13.5
f2_faintlim = np.float(sys.argv[2])

# global defaults
baddetect = 0.000001
ncmd = 1
smooth = 5
include = 0
exclude = 0
background1 = '-1 3 -1bg.dat\n'
background2 = '-1 1 -1\n'


# search ranges
imf = np.arange(-1., 3., 0.2)
bf = np.arange(0.0, 1.05, 0.1)
dav = np.arange(0, 1.1, 0.1)
#subs = ['ov{:.2f}'.format(o) for o in np.arange(0.3, 0.60, 0.05)]
subs = ['ov0.60']
timebins0 = np.arange(7.35, 9.41, 0.1)
timebins1 = np.arange(7.45, 9.51, 0.1)
tbins = len(timebins0)


shead="calcsfh='/Users/rosenfield/research/match2.5/bin/calcsfh'\n"
line1_2 = '%4.2f %4.2f %4.2f %4.2f %4.2f %4.2f\n' % (dmod0, dmod1, dmod_delta,
                                                     av0, av, av_delta)
line1_2 += ('%4.2f %4.2f %4.2f\n') % (zmin, zmax, z_delta)
footer = '%8.6f %8.6f\n%1i\n' % (baddetect, baddetect, ncmd)
footer += '%4.2f %4.2f %1i %4.2f %4.2f %s,%s\n' % (mbin, cbin, smooth, cmin,
                                                   cmax, filter1, filter2)
footer += '%4.2f %4.2f %s\n' % (f1_brightlim, f1_faintlim, filter1)
footer += '%4.2f %4.2f %s\n' % (f2_brightlim, f2_faintlim, filter2)
footer += '%1i %1i\n' % (include, exclude)
footer += '%2i\n' % (tbins)
footer += ''.join(['%4.2f %4.2f\n' % (timebins0[k], timebins1[k])
                   for k in range(len(timebins0))])
footer += background1
footer += background2

l = 0
nproc = 11
k = 0
sline = shead
ntot = len(imf) * len(bf) * len(dav) * len(subs) - 1
for i in range(len(imf)):
    for j in range(len(bf)):
        for m in dav:
            fmt = 'imf{2:0.1f}_bf{0:.1f}_dav{1:.1f}'.format(bf[j], m, imf[i])
            outparam = 'ngc2100_{}.param'.format(fmt)
            line = '%4.2f %s%4.2f ' % (imf[i], line1_2, bf[j])
            line += footer
            with open(outparam, 'w') as f:
                f.write(line)
            for sub in subs:
                l += 1
                fmt += '_{}'.format(sub)
                sline += '$calcsfh {0} ngc2100_F555W_F814W.match ngc2100_F555W_F814W.matchfake out_{1} -davy=0 -dav={2:.1f} -ssp -full -PARSEC -sub={3} > ssp_{1}.dat & \n'.format(outparam, fmt, m, sub)
                if l % nproc == 0 or l == ntot:
                    sline += 'wait\n'
                    #with open('script_{0}.sh'.format(k), 'w') as sf:
                    #    sf.write(sline)
                    #sline = shead
                    k += 1
with open('script.sh', 'w') as sf:
    sf.write(sline)
