import logging
import re

import numpy as np
from ades import adesutility
from ades.adesutility import applyPaddingAndJustification
from ades.psvtoxml import (
    DATA_RECORD_LINE,
    EMPTY_STATE,
    KEYWORD_RECORD_LINE,
    SECOND_HEADER_LINE,
    TOP_HEADER_LINE,
    stateTransitions,
)
from astropy.table import Table
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

allowedElementDict, requiredElementDict, psvFormatDict = adesutility.getAdesTables()

# psvFormatDict doesn't seem to match the achieved format... not sure why not - longer
# entries based on the data is one thing, but swapping justification is another.
# Turns out the justification is completely ignored in `xmltopsv.printDataDicts()` which
# calls `applyPaddingAndJustification()` with hardcoded 'L' values for everything....<sigh>
# We'll just define our own version here based on what is in the Astrometrica output files
# for optical observations and the tests in the ADES-Master repository
# psvFormatDict = {
#     'optical': [
#         ('permID', 7, 'L', 0),
#         ('provID', 11, 'L', 0),
#         ('artSat', 8, 'R', 0),
#         ('trkSub', 8, 'L', 0),
#         ('mode', 4, 'R', 0),
#         ('stn', 4, 'L', 0),
#         ('prog', 2, 'R', 0),
#         ('obsTime', 23, 'L', 0),
#         ('ra', 11, 'D', 3),
#         ('dec', 11, 'D', 4),
#         ('rmsRA', 5, 'D', 2),
#         ('rmsDec', 6, 'D', 2),
#         ('rmsCorr', 7, 'D', 3),
#         ('astCat', 8, 'L', 0),
#         ('mag', 5, 'D', 3),
#         ('rmsMag', 4, 'D', 2),
#         ('band', 3, 'R', 0),
#         ('photCat', 8, 'L', 0),
#         ('photAp', 4, 'D', 3),
#         ('logSNR', 4, 'D', 2),
#         ('seeing', 3, 'D', 2),
#         ('exp', 4, 'L', 0),
#         ('notes', 5, 'L', 0),
#         ('remarks', 0, 'L', 0),
#     ]
# }


# A lot of this ADES PSV parsing code is adapted from `psvtoxml.py` in the `ades` package
# but somewhat simplified for our purposes here and to avoid carrying lots of `globals` about

#
# create regular expression matching a string with no
# character other than [A-Za-z_]. The first character
# of all keywords must match this set.  That means
# we don't have any strange characters in keywords
#
allAlpha = re.compile('^[A-Za-z_]*$')


def parsePSVLine(line):
    """parsePSVLine classifies the PSV lines into:
    blank line:
      returns None
    TopHeader:
      returns (TOP_HEADER_LINE, [<name>, <reast of line>])
    SecondHeader:
      returns (SECOND_HEADER_LINE, [<name>, <reast of line>])
    KeywordRecord:
      returns (KEYWORD_RECORD_LINE, [<fields>, tag])
    DataRecord:
      returns (DATA_RECORD_LINE, [<fields>])
    """
    line = line.strip()  # strip leading and trailing whitespace

    if len(line) == 0:  # empty ines
        return None

    if line[:1] == '#':  # TopHeaderLine starts with '#' character
        return (TOP_HEADER_LINE, line[1:].split(None, 1))

    if line[:1] == '!':  # SecondHeaderLine starts with '!' character
        return (SECOND_HEADER_LINE, line[1:].split(None, 1))

        # pipe-separated fields with leading
        # and trailing whitespace removed
    fields = [a.strip() for a in line.split('|')]

    # Must be keyword header or data line
    # Classify using the requirement that
    # all xml element names must start with
    # letters or an underscore (like C identifiers)
    #
    # note obsTime, a required field, must start
    # with a digit, so all valid keyword and data
    # lines may be distingusished this way
    #
    # The above is subtle because it only works
    # since obsTime is alwyas required.  This is
    # a hard-coded requirement on the ades standard
    #

    # if fields[:1] == ["permID"]: # If so, must be header not data.  This is a less
    # subtle way of distinguishing, but sometimes the
    # permID fields are all blank.
    #

    firstChars = ''.join([x[:1] for x in fields])  # first character from all fields
    if allAlpha.match(firstChars):  # If so, must be header not data.
        #
        # first determine the type tag and append to fields
        #
        #
        # the tag is identified from elements in header
        #   It is identified by the fields it contains
        #   It must match all the required elements and
        #   have no extra elements for 'optical', 'offset',
        #   'occultation' or 'radar' -- these are the allowed
        #   under 'obsBlock'
        #
        #  Need to check if no match occurs (None flags this)
        #
        s = set(fields)
        tag = None
        for possible in allowedElementDict['ades']:
            #
            # obsBlock is in this list but will never match
            #
            # no extra but all required elements must in s
            #
            if s.issubset(allowedElementDict[possible]) and s.issuperset(requiredElementDict[possible]):
                tag = possible
        if not tag:
            raise RuntimeError('No matching header type')

        fields.append(tag)  # the last one is the tag

        return (KEYWORD_RECORD_LINE, fields)
    else:  # if not header record, must be data
        return (DATA_RECORD_LINE, fields)


def parsePSV(psvline, stack):
    """Parses a single line of a PSV format ADES file
    Takes, updates and returns the "stack" dictionary being built up from the
    PSV file (which in our case is a simpler dictionary as we're not building an XML)
    Modified from the original in `psvtoxml.py` in the `ades` package to:
    1) simplify the "stack" to just a dictionary which is passed in and updated
    2) remove the use of global variables except for state tracking
    3) add version checking for the first line
    4) doesn't have the same state transition complexity and checks as the original

    :param psvline: line from PSV file to be parsed
    :type psvline: str
    :param stack: The "stack" dictionary being built up from the PSV file.
    :type stack: dict
    :return: The "stack" dictionary being built up from the PSV file.
    :rtype: dict
    """

    global state
    global lineNumber
    global firstLine

    parsedLine = parsePSVLine(psvline)

    if not parsedLine:
        return None

    # first non-blank line must be "#version=2017" or "#version=2022"
    if firstLine:
        firstLine = False
        line = psvline.split('=')
        if len(line) != 2:
            raise RuntimeError("first line of PSV must specify version, e.g., '#version=2017'")
        if ''.join(line[0].split()) != '#version':
            raise RuntimeError("first line of PSV must specify version, e.g., '#version=2017'")
        version = line[1].strip()
        if version not in ['2017', '2022']:
            raise RuntimeError("PSV version must be '2017' or '2022'")
        stack['header']['version'] = version
        return stack

    record = parsedLine[0]
    nextState = stateTransitions[state][record][0]
    fields = parsedLine[1]
    # if state != 'ObsDataState':
    # print(state, record, nextState)
    # print(fields)
    if nextState == 'ObsContextState':
        if record == TOP_HEADER_LINE:
            stack['header'][fields[0]] = {}
            stack['currentHeader'] = fields[0]
        elif record == SECOND_HEADER_LINE:
            stack['header'][stack['currentHeader']][fields[0]] = fields[1]
        else:
            print('unexpected record type in ObsContextState:', record)
    elif nextState == 'FirstObsDataState':
        if record == KEYWORD_RECORD_LINE:
            stack['keywords'] = fields[:-1]  # all but last field (tag/observation type)
    elif nextState == 'ObsDataState':
        if record == DATA_RECORD_LINE:
            stack['observations'].append(fields)
        else:
            print('unexpected record type in ObsDataState:', record)
    state = nextState  # stateTransitions[state][record][1](state, nextState, record, fields)
    return stack


def read_psv(psvfile, psvencoding='UTF-8'):
    """Reads a PSV format ADES file

    :param psvfile: Filepath to PSV file
    :type psvfile: str
    :param psvencoding: PSV file encoding, defaults to "UTF-8"
    :type psvencoding: str, optional
    :return: Dictionary with 'header', 'keywords' and 'observations' keys
    :rtype: dict
    """

    global state
    state = EMPTY_STATE  # Initial State
    global firstLine
    firstLine = True
    stack = {'header': {}, 'observations': []}

    with open(psvfile, encoding=psvencoding) as f:
        lineNumber = 0
        for line in f:
            try:
                lineNumber += 1
                status = parsePSV(line.rstrip(), stack)
                if status is not None:
                    stack = status
            except RuntimeError as e:
                print(e)
                logger.error(f'Error parsing line {lineNumber} of {psvfile}: {e}')

    del stack['currentHeader']
    return stack


def convert_ades_to_table(ades_data):
    """Convert ADES data dictionary to a Table format.

    :param ades_data: ADES data dictionary with 'keywords' and 'observations'
    :type ades_data: dict
    :return: Table format with header and rows
    :rtype: astropy.table.Table
    """

    # Assemble dtypes for each column based on psvFormatDict (way harder than it should be as it's missing some columns
    # and has others that aren't needed in our case)
    dtypes = []
    skip_cols = ['artSat', 'prog', 'rmsCorr']
    for col in psvFormatDict['optical']:
        if col[0] == 'photCat' and 'fltr' in ades_data['keywords']:
            # psvFormatDict is missing an entry for 'fltr' (which is only available in ADES version=2022)
            # Insert it first if we have that column keyword
            dtypes.append('U3')
        elif col[0] == 'notes' and 'rmsFit' in ades_data['keywords'] and 'nStars' in ades_data['keywords']:
            # Insert dtype for missing rmsFit
            dtypes.append('f4')
            # Insert dtype for missing nStars
            dtypes.append('i4')
        elif col[0] in skip_cols and col[0] not in ades_data['keywords']:
            # print(f'Skipping {col[0]}')
            continue
        dtype = 'f8'
        if col[2] in ['L', 'R']:
            length = col[1]
            if length == 0:
                length = 300
            dtype = f'U{length}'
        dtypes.append(dtype)

    table = Table(rows=ades_data['observations'], names=ades_data['keywords'], dtype=dtypes)

    return table


def zero_aperture_extrapolation(ades_table, reference_ap=1.8, make_plots=False):
    """Perform the linear extrapolation to zero aperture size for RA and Dec
       for the passed ADES table `ades_table`. The values at the reference aperture
       size `reference_ap` (in arcsec) are used as the base values.
    XXX this needs to be checked for in the apertures avalilable in the table

    :param table: Table of ADES observations from convert_ades_to_table()
    :type table: astropy.table.Table
    :param reference_ap: Reference aperture size in arcseconds, defaults to 1.8
    :type reference_ap: float, optional
    :param make_plots: Whether to generate plots of the linear fits, defaults to False
    :type make_plots: bool, optional
    :return: Dictionary of derived zero-aperture RA and Dec values for each obsTime
    :rtype: dict
    """

    from astropy import table

    marker_col = 'k'
    zaa_col = '#1f77b4'

    derived_zaa = {}
    unique_dts = table.unique(ades_table, keys='obsTime')
    for dt in unique_dts['obsTime']:
        mask = ades_table['obsTime'] == dt
        sub_table = ades_table[mask]
        # Determine target name; assume it's the same for all entries at this obsTime
        # work along the permID, provID, trkSub columns to find the first non-empty value
        target_name = 'Unknown'
        for col in ['permID', 'provID', 'trkSub']:
            if col in sub_table.colnames:
                non_empty = sub_table[col][sub_table[col] != '']
                if len(non_empty) > 0:
                    target_name = non_empty[0]
                    break

        if 'ra' in sub_table.colnames and 'dec' in sub_table.colnames:
            apertures = sub_table['photAp']
            ras = sub_table['ra'] - sub_table['ra'][apertures == reference_ap][0]
            cosdec = np.cos(np.deg2rad(sub_table['dec'][apertures == reference_ap][0]))
            ras = ras * cosdec * 3600  # Convert to arcseconds
            ra_errs = sub_table['rmsRA'] * cosdec
            decs = sub_table['dec'] - sub_table['dec'][apertures == reference_ap][0]
            decs = decs * 3600  # Convert to arcseconds
            dec_errs = sub_table['rmsDec']
            if len(apertures) >= 2:
                # Perform linear fit to ra vs. photAp
                coeffs_ra = np.polyfit(apertures, ras, 1)
                # Extrapolate to zero aperture size
                delta_zero_ap_ra = np.polyval(coeffs_ra, 0)
                zero_ap_ra = sub_table['ra'][apertures == reference_ap][0] + (delta_zero_ap_ra / 3600.0) / cosdec
                print(f'Extrapolated RA for obsTime {dt}: {zero_ap_ra:.7f}')
                # Perform linear fit to dec vs. photAp
                coeffs_dec = np.polyfit(apertures, decs, 1)
                # Extrapolate to zero aperture size
                delta_zero_ap_dec = np.polyval(coeffs_dec, 0)
                zero_ap_dec = sub_table['dec'][apertures == reference_ap][0] + (delta_zero_ap_dec / 3600.0)
                print(f'Extrapolated Dec for obsTime {dt}: {zero_ap_dec:+.7f}')
                # Make summary plots of the extrapolated values
                if make_plots is True:
                    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150)
                    # Reserve space on the right of the figure for the external legend
                    fig.subplots_adjust(right=0.78)
                    fig.suptitle(f'{target_name} -- {dt} - Linear Fit')
                    xmax = np.ceil(apertures.max())
                    x = [0.0, apertures.min(), xmax]
                    coeffs_arr = np.polyval(coeffs_ra, x)

                    ax1.plot(x[1:], coeffs_arr[1:], color=marker_col, label='RA Fit')
                    ax1.scatter(0.0, delta_zero_ap_ra, marker='o', color=zaa_col, label='0 Aperture\nExtrapolation')
                    ax1.plot(x[0:2], coeffs_arr[0:2], color=zaa_col, linestyle='--')
                    ax1.errorbar(
                        apertures,
                        ras,
                        yerr=ra_errs,
                        marker='d',
                        color=marker_col,
                        linestyle='',
                        label='Included RA Data',
                    )
                    # Remove the offset from both x and y axes, turn on minor ticks for y axis
                    ax1.ticklabel_format(useOffset=False, axis='both')
                    ax1.yaxis.minorticks_on()
                    ax1.set_ylabel(r'$\Delta \text{RA}*\cos(\text{Dec})$ (arcsec)')
                    ax1.set_title(f'RA: {zero_ap_ra:.7f}' + r'$^\circ$')
                    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

                    # Same again for Dec
                    coeffs_arr = np.polyval(coeffs_dec, x)
                    ax2.plot(x[1:], coeffs_arr[1:], color=marker_col, label='Dec Fit')
                    ax2.scatter(0.0, delta_zero_ap_dec, marker='o', color=zaa_col, label='0 Aperture\nExtrapolation')
                    ax2.plot(x[0:2], coeffs_arr[0:2], color=zaa_col, linestyle='--')
                    ax2.errorbar(
                        apertures,
                        decs,
                        yerr=dec_errs,
                        marker='s',
                        markersize=8,
                        color=marker_col,
                        linestyle='',
                        label='Included Dec Data',
                    )
                    ax2.set_title(f'Dec: {zero_ap_dec:+.7f}' + r'$^\circ$')
                    # Remove the offset from both x and y axes, turn on minor ticks for y axis
                    ax2.ticklabel_format(useOffset=False, axis='both')
                    ax2.yaxis.minorticks_on()
                    ax2.set_xlabel('Aperture Size (arcsec)')
                    ax2.set_ylabel(r'$\Delta \text{Dec}$ (arcsec)')
                    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
                    fig.tight_layout()
                derived_zaa[str(dt)] = {
                    'zero_ap_ra': zero_ap_ra,
                    'zero_ap_dec': zero_ap_dec,
                    'delta_zero_ap_ra': delta_zero_ap_ra,
                    'delta_zero_ap_dec': delta_zero_ap_dec,
                }
    return derived_zaa


def write_psv(ades_data, zaa_ra_dec, output_psv_file, reference_ap=1.8, psvencoding='UTF-8'):
    """Write ADES data dictionary to a PSV format ADES file, including zero-aperture RA and Dec values.

    :param ades_data: ADES data dictionary with 'keywords' and 'observations'
    :type ades_data: dict
    :param zaa_ra_dec: List of tuples with zero-aperture RA and Dec values
    :type zaa_ra_dec: list of tuples
    :param output_psv_file: Filepath to output PSV file
    :type output_psv_file: str
    :param reference_ap: Reference aperture size in arcseconds, defaults to 1.8
    :type reference_ap: float, optional
    :param psvencoding: PSV file encoding, defaults to "UTF-8"
    :type psvencoding: str, optional
    """

    with open(output_psv_file, 'w', encoding=psvencoding) as f:
        # Write version line
        f.write('# version=2022\n')
        # Write header lines
        for key, value in ades_data['header'].items():
            if key == 'version':
                # version is special case, already written
                continue
            elif isinstance(value, dict):
                f.write(f'# {key}\n')
                for subkey, subvalue in value.items():
                    f.write(f'! {subkey} {subvalue}\n')
            else:
                f.write(f'# {key} {value}\n')

        # Write keyword line
        ## Turn psvFormatDict into an actual dict of colname: (length, type)
        psvList = []
        headerInfo = []
        headerDict = {}
        cols = []
        for j in psvFormatDict['optical']:
            if j[0] in ades_data['keywords']:
                psvList.append(j[0])  # keep for adding other elements
                headerItem = [j[0], int(j[1]), j[2], j[3], False]
                headerInfo.append(headerItem)
                headerDict[j[0]] = headerItem

        #
        # add in the rest of the allowed elements between notes and remarks
        #
        for j in allowedElementDict['optical']:  # allowed elements
            if j not in psvList:  # not already in list
                #
                # Insert before "remarks", which is the last item
                #
                headerItem = [j, len(j), 'R', 0, False]
                headerInfo.insert(-1, headerItem)
                headerDict[j] = headerItem

        #
        # adjust lengths if header doesn't fit (not sure this does anything useful)
        #
        for headerItem in headerInfo:
            if len(headerItem[0]) > headerItem[1]:  # too wide
                headerItem[1] = len(headerItem[0])
            if headerItem[0] == 'trkSub':  # special case
                headerItem[1] = max(headerItem[1], 8)
            elif headerItem[0] == 'rmsRA':
                headerItem[1] = 5
            elif headerItem[0] == 'rmsDec':
                headerItem[1] = 6
            elif headerItem[0] == 'ra' or headerItem[0] == 'dec':
                headerItem[1] = 11
            elif headerItem[0] == 'astCat' or headerItem[0] == 'photCat':
                headerItem[1] = 8
            headerDict[headerItem[0]] = headerItem

        for i in ades_data['keywords']:
            if i in headerDict:
                headerItem = headerDict[i]
                # Apparently justification is a "pirate guidelines" and is always 'L' in the header...
                col, w, dposnew = applyPaddingAndJustification(headerItem[0], headerItem[1], 'L', 0)
                cols.append(col)

        f.write('|'.join(cols) + '\n')
        index_photAp = ades_data['keywords'].index('photAp')
        # Write data lines with zero-aperture RA and Dec appended
        for observation in ades_data['observations']:
            # Skip line if photAp is not the reference aperture size
            if float(observation[index_photAp]) != reference_ap:
                continue
            obsTime = None
            obsline = []
            for col, value in zip(ades_data['keywords'], observation):
                if col in headerDict:
                    headerItem = headerDict[col]
                    if col == 'obsTime':
                        obsTime = value
                    elif col == 'ra':
                        value = f"{zaa_ra_dec[obsTime]['zero_ap_ra']:010.6f}"
                    elif col == 'dec':
                        value = f"{zaa_ra_dec[obsTime]['zero_ap_dec']:+09.6f}"
                    elif col == 'notes':
                        value = 'e' + value  # mark as extrapolated
                    # print(value, type(value), headerItem[1], type(headerItem[1]), headerItem[2], \
                    # type(headerItem[2]),  headerItem[3], type(headerItem[3]))
                    col_str, w, dposnew = applyPaddingAndJustification(
                        value, headerItem[1], headerItem[2], headerItem[3]
                    )
                    obsline.append(col_str)
            f.write('|'.join(obsline).rstrip() + '\n')


def perform_zero_aperture_extrapolation_on_psv(
    input_psv_file, output_psv_file, reference_ap=1.8, make_plots=False, psvencoding='UTF-8'
):
    """Perform zero-aperture extrapolation on a PSV format ADES file and write the results to a new PSV file.

    :param input_psv_file: Filepath to input PSV file
    :type input_psv_file: str
    :param output_psv_file: Filepath to output PSV file
    :type output_psv_file: str
    :param reference_ap: Reference aperture size in arcseconds, defaults to 1.8
    :type reference_ap: float, optional
    :param make_plots: Whether to generate plots of the linear fits, defaults to False
    :type make_plots: bool, optional
    :param psvencoding: PSV file encoding, defaults to "UTF-8"
    :type psvencoding: str, optional
    """

    ades_data = read_psv(input_psv_file, psvencoding=psvencoding)
    ades_table = convert_ades_to_table(ades_data)
    zaa_ra_dec = zero_aperture_extrapolation(ades_table, reference_ap=reference_ap, make_plots=make_plots)
    write_psv(ades_data, zaa_ra_dec, output_psv_file, reference_ap=reference_ap, psvencoding=psvencoding)
