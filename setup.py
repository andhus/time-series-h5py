from setuptools import setup, find_packages

VERSION = '0.0.1'

setup(
    name='time-series-h5py',
    version=VERSION,
    description=(
        'Abstraction layer on top of h5py for storing time-indexed Pandas data'
    ),
    url='https://github.com/andhus/time-series-h5py',
    license='MIT',
    install_requires=[
        'numpy>=1.13.0',
        'h5py>=2.7.0',
        'pandas>=0.20.0'
    ],
    extras_require={},
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
)
