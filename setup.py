#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup, find_packages

setup(name='pyket', 
    version='0.0.1',
    description='VMC framework for Tensorflow',
    url='https://github.com/HUJI-Deep/AutoregressiveQuantumModel',
    author='Noam Wies',
    author_email='noam.wies@mail.huji.ac.il',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,

    install_requires=[
        # todo support also gpu
        'tensorflow>=1.10',
        'tqdm>=4.31.1'
    ],

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    project_urls={
        # 'Documentation': 'https://Pyket.readthedocs.io/',
        # 'Changelog': 'https://Pyket.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/noamwies/Pyket/issues',
    },
    )