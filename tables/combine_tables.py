from astropy.coordinates import SkyCoord
import astropy.units as u

data = read_hlacsv('tables/baumgardt13_data.csv', raunit='hms')

radec = SkyCoord(ra=data['RA'], dec=data['Dec'], unit=(u.hourangle, u.deg))

radec = np.array([r.split() for r in radec.to_string('decimal')], dtype=float)

tab = Table.read('/Users/rosenfield/Downloads/lmc_fields_MAST_Crossmatch_All_Observations.csv', header_start=2)
inds, = np.nonzero(np.abs(np.array(tab['Proposal ID'][2:], dtype=float) - np.array(tab['propid'][2:], dtype=float))==0)
tab2 = tab[2:][inds]
