
def test_minimal():

    import bornagain
    from bornagain import detector
    from bornagain import source
    from bornagain import utils
    from bornagain import units

    assert bornagain is not None


if __name__ == '__main__':
    test_minimal()
    print('ran tests')
