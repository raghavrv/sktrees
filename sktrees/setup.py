from numpy.distutils.misc_util import Configuration
from sklearn._build_utils import maybe_cythonize_extensions
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    config = Configuration('sktrees', parent_package, top_path)
    config.add_subpackage('fast_tree')
    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
