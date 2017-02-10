def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    for old, new in dic.items():
        text = text.replace(old, new)
    return text


def center_from_simbad(target):
    """Query Simbad for the coordinates of a target."""
    from astroquery.simbad import Simbad
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    def sstr(attr):
        """
        Strip the value from a simbad table string
        e.g.
        >>> str(q['RA'])
        >>> '     RA     \n  "h:m:s"   \n------------\n04 52 25.040'
        >>> sstr(q['RA'])
        >>> '04 52 25.040'
        """
        return str(attr).split('\n')[-1]

    qry = Simbad.query_object(target)

    if qry is None:
        print('Error, can not query simbad for {}'.format(target))
        return np.nan, np.nan

    radec = SkyCoord(ra=sstr(qry['RA']), dec=sstr(qry['DEC']),
                     unit=(u.hourangle, u.deg))
    return radec.ra.value, radec.dec.value


def fixnames(data):
    """
    Mark empty names with NONAME not np.nan.
    Also remove spaces and "Cl" and []
    """
    for i in range(len(data.SimbadName)):
        try:
            float(data.loc[i].SimbadName)
            data.SimbadName.iloc[i] = 'NONAME'
        except:
            pass

    repd = {' ': '', '[': '', ']': '', 'Cl': ''}
    names = [replace_all(l, repd) for l in data.SimbadName]
    data.SimbadName = names
    return data
