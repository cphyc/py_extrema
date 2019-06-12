# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# with open('Readme.md') as f:
#     readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='py_extrema',
    version='0.0.2',
    description='Find extrema using python.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    author='Corentin Cadiou',
    author_email='contact@cphyc.me',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'numba',
        'unyt',
        'scipy',
        'pandas',
        'pyfftw',
        'tqdm',
        'numexpr'
    ],
    include_package_data=True
)
