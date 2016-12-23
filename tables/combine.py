import pandas as pd
from scipy.spatial import KDTree
import numpy as np

    
def nearest(t1, t2, tol=0.1):
    """
    A bit flexible about duplicates...
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    
    dists = np.array([])
    ix2s = np.array([], dtype=int)
    
    for ix1 in range(len(t1)):
        dif = np.abs(t1[ix1] - t2)
        ix2, = np.nonzero(dif < tol)
        if len(ix2) == 0:
            ix2 = np.nan
            dist = np.nan
        else:
            dist = np.min(dif)
            ix2 = np.argmin(dif)

        dists = np.append(dists, dist)
        ix2s = np.append(ix2s, ix2)        
    return dists, ix2s

hstmc = pd.read_csv('HSTimagePointings.csv', header=0)
# Same instruments
hstmc = hstmc.iloc[np.where((hstmc['instrument'] == 'ACS/WFC') |
                          (hstmc['instrument'] == 'WFC3/UVIS') |
                          (hstmc['instrument'] == 'WFC3/IR'))]

# Near LMC/SMC
hstmc = hstmc.iloc[np.where((hstmc['s_ra'] > 0.95) &
                            (hstmc['s_ra'] < 96.5) &
                            (hstmc['s_dec'] > -90) &
                            (hstmc['s_dec'] < -59))].copy(deep=True)

mcleg = pd.read_csv('table_flt.csv', header=1)

s_region = np.zeros(len(mcleg), dtype='|S1024')
haves = 0
havenots = 0
nomatch = pd.DataFrame()

filts = np.unique(np.concatenate([mcleg['filter1'], mcleg['filter2']]))

for filt in filts:
    if filt == '...' or filt.startswith('CLEAR') or filt.endswith('N') \
        or filt.endswith('M'):
        continue
    imfilt, = np.nonzero((np.array(mcleg['filter2'], dtype=str) == filt) |
                        (np.array(mcleg['filter1'], dtype=str) == filt))

    ihfilt, = np.nonzero((np.array(hstmc['filters'], dtype=str) == filt))
    
    mclegf = mcleg.iloc[imfilt].copy(deep=True)
    hstmcf = hstmc.iloc[ihfilt].copy(deep=True)
    dist, hidx =  nearest(mclegf['t_min'], hstmcf['t_min'], tol=1.)
    
    s_region[imfilt[np.isfinite(hidx)]] = hstmc['s_region'].iloc[hidx[np.isfinite(hidx)]]
    nomatch = nomatch.append(mclegf.iloc[np.isnan(hidx)], ignore_index=True)
    have = sum(np.isfinite(dist))
    havenot = sum(np.isnan(dist))
    print('{} have {}, missing {}'.format(filt, have, havenot))
    haves += have
    havenots += havenot
print('total: {} missing: {}'.format(haves, havenots))
mcleg['s_region'] = s_region
mcleg.to_csv('table_flt_sregion.csv', index=False)

"""

# Same filters
filts = np.unique(np.concatenate([mcleg['filter1'], mcleg['filter2']]))
ifilt = [i for i, f in enumerate(hstmc['filters']) if f in filts]

# Near LMC/SMC
hstmc = hstmc.iloc[ifilt].copy(deep=True)
hstmc = hstmc.iloc[np.where((hstmc['s_ra'] > 0.95) &
                            (hstmc['s_ra'] < 96.5) &
                            (hstmc['s_dec'] > -90) &
                            (hstmc['s_dec'] < -59))].copy(deep=True)
                            
# Drop parallels (same s_regions)
hstmc = hstmc.drop_duplicates(subset='t_min')
                            
                            
                            
                            
"""
