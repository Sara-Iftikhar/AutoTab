
from setuptools import setup

import os

long_desc="automl",

fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(

    name='automl',

    version="0.1",

    description='optimization of ML pipeline using hierarchical optimization method',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/Sara-Iftikhar/automl',

    author='Sara Iftikhar',
    author_email='sara.rwpk@gmail.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    packages=['automl'],

    install_requires=[
        #'ai4water[ml_hpo]>=1.0b6',
        'h5py',
    ],
    extras_require={
        'all': ["ai4water[ml_hpo]>=1.0b6", "h5py"],
    }
)