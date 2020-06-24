#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages

import acc

setup(
    name='acc',
    version=acc.__version__,
    packages=find_packages(),
    author="Yannis Flet-Berliac & Johan Ferret",
    author_email="yannisfbe@gmail.com",
    description="ACC RL agent",
    install_requires=[
        'cloudpickle',
        #'cv2',
        'gym',
        'gym-minigrid',
        'matplotlib',
        #'mpi4py',
        'numpy',
        'optuna',
        'pandas',
        #'pybullet',
        'scipy',
        'seaborn',
        'slimevolleygym',
        'tensorflow',
        #'vizdoomgym @ git+ssh://git@github.com/shakenes/vizdoomgym@master#egg=vizdoomgym',
    ],
    #dependency_links=[
    #    'https://github.com/shakenes/vizdoomgym',
    #],
    include_package_data=True,
    url='http://github.com/confituredefigues/acc',
    classifiers=[
        "Programming Language :: Python",
    ],
    entry_points = {
    },
    license="MIT",
)