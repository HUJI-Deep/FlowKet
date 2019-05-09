from setuptools import setup, find_packages

setup(name='pyket', 
    version='0.0.1',
    description='VMC framework for Tensorflow',
    url='https://github.com/HUJI-Deep/AutoregressiveQuantumModel',
    author='Noam Wies',
    author_email='noam.wies@mail.huji.ac.il',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # todo support also gpu
        'tensorflow>=1.10',
        'tensorflow-addons>=0.3.1',
        'tqdm>=4.31.1'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False)