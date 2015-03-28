from setuptools import setup, find_packages  # Always prefer setuptools over distutils
import codecs
import os

import rppy

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rppy',
    version='0.0.1',
    url='https://github.com/shear/RPpy',
    license='GNU General Public License v2 (GPLv2)',
    author='Sean M. Contenti',
    author_email='sean.contenti@gmail.com',
    description='A Python Rock Physics library',
    long_description=long_description,
    packages=['rppy'],
    platforms='any',
    tests_require['pytest'],
    install_requires=['numpy'],
    keywords='geophysics','rock physics',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
    ],
)