#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='rppy',
    version='0.1.0',
    description="A geophysical library for Python",
    long_description=readme + '\n\n' + history,
    author="Sean Contenti",
    author_email='sean.contenti@gmail.com',
    url='https://github.com/shear/rppy',
    packages=[
        'rppy',
    ],
    package_dir={'rppy':
                 'rppy'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='rppy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
