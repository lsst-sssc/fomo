import fomo


def test_version():
    """Check to see that we can get the package version"""
    assert fomo.__version__ is not None
