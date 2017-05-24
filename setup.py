import os

import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from setuptools import find_packages


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
    sklearn_matches = False

    try:
        import sklearn
    except ImportError:
        pass
    else:
        if sklearn.__version__ == '0.19.dev0':
            sklearn_matches = True

    if not sklearn_matches:
        raise ImportError("This sktrees version will patch only onto "
                          "sklearn==0.19.dev0")

    setup(name='sktrees',
          maintainer='Raghav RV',
          maintainer_email='rvraghav93@gmail.com',
          description="Unofficial bleeding edge updates to sklearn's tree models",
          packages=find_packages(),
          license="new BSD",
          url="https://sktrees.github.io",
          version=VERSION,
          long_description=open('README.md').read(),
          install_requires=["scikit-learn==0.19.dev0"],
          zip_safe=False,  # the package can run out of an .egg file
          include_package_data=True,
          configuration=configuration
          )


if __name__ == "__main__":
    setup_package()
