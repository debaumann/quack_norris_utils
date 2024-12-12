# quack_norris_utils/setup.py
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['quack_norris_utils'],
    package_dir={'': '.'}
)

setup(**d)