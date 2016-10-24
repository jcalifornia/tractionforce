#!/usr/bin/env python

import os
from setuptools import find_packages, setup


def extract_version():
    """Return ggplot.__version__ without importing ggplot.

    Extracts version from ggplot/__init__.py
    without importing ggplot, which requires dependencies to be installed.
    """
    with open('tractionforce/__init__.py') as fd:
        ns = {}
        for line in fd:
            if line.startswith('__version__'):
                exec(line.strip(), ns)
                return ns['__version__']


setup(
    name="tractionforce",
    version=extract_version(),
    author="Josh Chang",
    author_email="josh.chang@nih.gov",
    packages=find_packages(),
    package_dir={ "tractionforce": "tractionforce" },
    package_data={
        "tractionforce": [
            "data/*.txt"
        ]
    },
    description="tractionforce",
    long_description=open("README.md").read(),
    install_requires=[
        "cvxopt",
        "cvxpy",
        "matplotlib",
        "scipy",
        "numpy"
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5'
    ],
    scripts=[
        'bin/reconstruction.py'
    ],
    zip_safe=False
)
