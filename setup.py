#!/usr/bin/env python

import os

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()


def locate(*names):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), *names))


SOURCE = locate('htm_source')
PACKAGES_REQUIRED = find_packages(SOURCE)
PACKAGES_REQUIRED = [os.path.join(SOURCE, x) for x in PACKAGES_REQUIRED]
if 'htm_source' not in PACKAGES_REQUIRED:
    PACKAGES_REQUIRED.insert(0, 'htm_source')

print(f"SOURCE = {SOURCE}")
print(f"PACKAGES_REQUIRED...")
for pr in sorted(PACKAGES_REQUIRED):
    print(f"  --> {pr}\n")

INSTALL_REQUIRES = [
    'pandas==1.2.4',
    'pip>=19.3.1',
    'PyYAML==6.0',
    'wheel>=0.33.6',
    'cmake>=3.14',  # >=3.7, >=3.14 needed for MSVC 2019
    ## for python bindings (in /bindings/py/)
    'numpy>=1.15',
    'pytest>=4.6.5',  # 4.6.x series is last to support python2, once py2 dropped, we can switch to 5.x
    ## for python code (in /py/)
    'hexy>=1.4.3',  # for grid cell encoder
    'mock>=1.0.1',  # for anomaly likelihood test
    'prettytable>=0.7.2',  # for monitor-mixin in htm.advanced (+its tests)
]

authors = "Sam Heiserman"
author_email = "sheiser1@binghamton.edu"

setup(
    name='htm_source',
    version='0.0.72',
    description='HTM Stream - Rapid ML prototyping tool for HTM anomaly detection on numeric time series',
    long_description=readme,
    license='MIT',
    author=authors,
    author_email=author_email,
    url='https://github.com/gotham29/htm_streamer',
    project_urls={
        "Bug Tracker": "https://github.com/gotham29/htm_streamer/issues"
    },
    packages=PACKAGES_REQUIRED,
    install_requires=INSTALL_REQUIRES,
    keywords=["pypi", "htm_source"],
)
