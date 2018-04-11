# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# with open('Readme.md') as f:
#     readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='py_extrema',
    version='0.0.1',
    description='Find extrema using python.',
    classifiers=[
        'Development status :: 1 - Alpha',
        'License :: CC-By-SA2.0',
        'Programming Language :: Python',
        'Topic :: Gaussian Random Field'
    ],
    author='Corentin Cadiou',
    author_email='contact@cphyc.me',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'numba'
    ],
    include_package_data=True
)
