import os
from distutils.core import setup

# The version will be the same as the scikit-learn version on which this can
# be patched
VERSION = "0.19.dev0"


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('sktrees')

    return config


def setup_package():
    metadata = dict(name='sktrees',
                    maintainer='Raghav RV',
                    maintainer_email='rvraghav93@gmail.com',
                    description=("Unofficial bleeding edge updates to sklearn"
                                 "'s tree models"),
                    license="new BSD",
                    url="https://sktrees.github.io",
                    version=VERSION,
                    long_description=open('README.md').read())

    sklearn_matches = False

    try:
        import sklearn
    except ImportError:
        pass
    else:
        if sklearn.__version__ >= '0.19.dev0':
            sklearn_matches = True

    if not sklearn_matches:
        raise ImportError("This sktrees version will patch only onto "
                          "sklearn>=0.19.dev0")

    from numpy.distutils.core import setup

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
