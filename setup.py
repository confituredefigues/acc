#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='acc',
    version="0.1.0",
    author="Yannis Flet-Berliac & Johan Ferret",
    author_email="fletberliac@gmail.com",
    description="ACC RL agent",
    install_requires=[
        'cloudpickle==1.2.2',
        'opencv-python',
        'gym',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy==1.1.0',
        'seaborn',
        'tensorflow==1.14.0',
        'pybullet',
        # 'gym-minigrid@git+https://github.com/confituredefigues/acc@develop#egg=gym-minigrid&subdirectory=gym-minigrid',
        'slimevolleygym@git+https://github.com/hardmaru/slimevolleygym@master#egg=slimevolleygym',
        # 'vizdoomgym@git+https://github.com/shakenes/vizdoomgym@master#egg=vizdoomgym',
        # 'optuna',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    url='http://github.com/confituredefigues/acc',
    classifiers=[
        "Programming Language :: Python",
    ],
    entry_points={
        "distutils.commands": ["compile = package.commands:Compile"]
    },
    license="MIT",
)

# If "ERROR: Failed building wheel for vizdoom", run:
# sudo apt-get install cmake libboost-all-dev libgtk2.0-dev libsdl2-dev python-numpy
