import os
from os.path import join
import warnings


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sktrees', parent_package, top_path)

    config.add_subpackage('fast_tree')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
