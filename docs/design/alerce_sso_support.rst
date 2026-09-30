Alerce Solar System Object Support
==================================

This document records research into what solar system object (SSO) support is
available to FOMO from the Alerce broker's Rubin/LSST alert processing, how it
compares to the JPL SBDB and MPC Explorer ingestion FOMO already has, and what
would need to change before Alerce becomes a useful target source.

Package Landscape
-----------------

Alerce is supporrted in the base tom_toolkit package, through the `alerce` subpackage. This is provided by the same `alerce_client` that is `pip`-installed at the top of the Alerce example notebook <https://github.com/alercebroker/usecases/blob/1f52178c836d21ef70086747cd3d07848a721762/notebooks/LSST/ALeRCE_LSST_SSO.ipynb>.

Initialization of the `alerce_client` is done with a `from alerce.core import Alerce` statement, and then an `alerce = Alerce()` call. The `alerce` object has a `query` method that can be used to query the Alerce database for alerts.

Notebook doesn't actually use the `alerce` object for the queries, but instead uses the TAP service through `pyvo` (the notebook does switch to using the `alerce` object for retrieving the image stamps which might be later functionality that FOMO would want to use). The TAP service is queried with a `pyvo.dal.TAPService` object, which is initialized with the Alerce TAP URL. The `search` method of the `TAPService` object is then used to execute an ADQL query against the Alerce database.

Attempting to replicate the notebook's query for a specific object through the `alerce` object results in apparent 500 server errors::

    from alerce.core import Alerce
    alerce = Alerce()

    lsst_query_parameters = {
        'classifiers': [],
        'format': 'json',
        'oid': '21163607367496779',
        'survey': 'lsst'
    }
    alerce.query_object(**lsst_query_parameters)

This produces the following error::

    File ~/venv/fomo_venv/lib/python3.12/site-packages/alerce/exceptions.py:23, in handle_error(response, response_format)
     20 code = response.status_code
     21 data = error
     23 raise codes.get(code, APIError)(
     24     message=message, code=code, data=data, response=response
     25 )

    APIError: {'Error code': 500, 'Message': 'An error occurred', 'Data': {}}

Created a Community Post <https://community.lsst.org/t/issues-querying-lsst-ssos-through-alerce-client/12428>
Someone also posted an issue on the Alerce GitHub repo <https://github.com/alercebroker/alerce_client/issues/72>) on April 8th - no response yet...
