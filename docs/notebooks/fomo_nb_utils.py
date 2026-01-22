import astropy.units as u
from astropy.coordinates import SkyCoord


def special_locations(skycoords=False):
    """Special locations for Rubin. Adapted and condensed into one routine from
    the originals in `rubin_scheduler.utils`

    Parameters
    ----------
    skycoords : `bool`
        Return locations as astropy.SkyCoords. If False, returns
        as a tuple of floats. Default False.
    """

    # The DDF locations here are from Neil Brandt's white paper
    # submitted in response to the 2018 Call for White Papers on observing
    # strategy
    # Document-30468 -- AGN-DDF-WP-02.pdf
    # The locations are chosen based on existing multi-wavelength
    # coverage, plus an offset to avoid the bright star Mira near XMM-LSS

    result = {}
    result['ELAISS1'] = SkyCoord('00:37:48 −44:01:30', unit=(u.hourangle, u.deg), frame='icrs')
    result['XMM_LSS'] = SkyCoord('02:22:18  −04:49:00', unit=(u.hourangle, u.deg), frame='icrs')
    result['ECDFS'] = SkyCoord('03:31:55  −28:07:00', unit=(u.hourangle, u.deg), frame='icrs')
    result['COSMOS'] = SkyCoord('10:00:26  +02:14:01', unit=(u.hourangle, u.deg), frame='icrs')
    result['EDFS_a'] = SkyCoord(ra=58.90 * u.deg, dec=-49.32 * u.deg, frame='icrs')
    result['EDFS_b'] = SkyCoord(ra=63.60 * u.deg, dec=-47.60 * u.deg, frame='icrs')
    result['Roman_bulge_location'] = SkyCoord(268.708 * u.deg, -28.975 * u.deg, frame='icrs')

    if not skycoords:
        # replace SkyCoord with ra/deg tuple in degrees
        result = dict([(key, (result[key].ra.deg, result[key].dec.deg)) for key in result])

    return result
