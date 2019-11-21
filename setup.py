#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os import path
from os.path import basename, splitext

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='flowket',
      version='0.2.1',
      description='VMC framework for Tensorflow',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/HUJI-Deep/FlowKet',
      author='Noam Wies',
      author_email='noam.wies@mail.huji.ac.il',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
      zip_safe=False,

      install_requires=[
          'tqdm>=4.31.1', 'numpy>=1.16', 'networkx>=2.3'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      project_urls={
          # 'Documentation': 'https://pyket.readthedocs.io/',
          # 'Changelog': 'https://pyket.readthedocs.io/en/latest/changelog.html',
          'Issue Tracker': 'https://github.com/HUJI-Deep/FlowKet/issues',
      },
      )
