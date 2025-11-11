import logging
import re

from ades import adesutility
from ades.psvtoxml import (
    DATA_RECORD_LINE,
    EMPTY_STATE,
    KEYWORD_RECORD_LINE,
    SECOND_HEADER_LINE,
    TOP_HEADER_LINE,
    stateTransitions,
)

logger = logging.getLogger(__name__)

allowedElementDict, requiredElementDict, _ = adesutility.getAdesTables()

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

    :param psvline: line from PSV file to be parsed
    :type psvline: str
    :return: _description_
    :rtype: _type_
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

    :param psvfile: _description_
    :type psvfile: _type_
    :param psvencoding: PSV file encoding, defaults to "UTF-8"
    :type psvencoding: str, optional
    :return: _description_
    :rtype: _type_
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
