#!/usr/bin/env python

import os

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()


def locate(*names):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), *names))


SOURCE = locate('src')
PACKAGES_REQUIRED = find_packages(SOURCE)
PACKAGES_REQUIRED = [os.path.join(SOURCE, x) for x in PACKAGES_REQUIRED]

INSTALL_REQUIRES = [
    'joblib==0.17.0',
    'nltk==3.5',
    'numpy==1.19.4',
    'scikit-plot==0.3.7',
    'scikit-learn==0.24.2',
    'pandas==1.2.4',
    'pytest==6.1.2',
    'python-dotenv==0.15.0',
    'requests==2.25.0',
    'matplotlib==3.3.3',
    'schedule==0.6.0',
    'xgboost==1.3.1',
    'category_encoders==2.2.2',
    'pep8==1.7.1',
    'Flask==1.1.0',
    'Flask-API==2.0',
    'flask-cors==3.0.9',
    'Jinja2==2.11.1',
    'pyotp>=2.3.0',
    'imbalanced-learn==0.8.0',
    'openpyxl==3.0.7',
    'scikit-optimize==0.8.1',
    'python-dateutil==2.8.1'
]

authors = "Bruce Li, Chunling Zhang, Hongya Lu, Ruslan Lapin, Sam Heiserman, Steven Lee, Vedant Dhandhania"
author_email = "bruce_li2@apple.com, chunling_zhang@apple.com, hongya_lu@apple.com, rlapin@apple.com, heiserman@apple.com, lsteven@apple.com, vdhandhania@apple.com"

setup(
    name='bootstrap',
    version='0.1.26',
    description='DS Bootstrap - ML prototyping tool for OpsML team',
    long_description=readme,
    license='Apple Internal',
    author=authors,
    author_email=author_email,
    url='https://github.pie.apple.com/opsml/ds-bootstrap',
    packages=PACKAGES_REQUIRED,
    install_requires=INSTALL_REQUIRES,
    keywords="bootstrap opsml",
)
