#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requires = [
    'pandas>=1.3.0,<1.4',
    'scikit-learn>=1.0.2,<1.1',
    'scanpy>=1.8.2,<1.9',
    'scipy>=1.7.3,<1.8',

    # Flask
    'flask>=1.1.0,<1.2',
    'flask-restful>=0.3.8,<0.4.0',
    'flask-cors>=3.0,<3.1',
    'flasgger==0.9.5',
]

tests_requires = [
    'pytest==6.2.4',
    'pytest-cov>=2.6.0',
    'coverage==4.5.4',
]

dev_requires = [
    # general
    'pip>=19.2.3',
    'bump2version>=0.5.11',
    'wheel==0.33.6',
    'watchdog==0.9.0',
    'jupyter>=1.0.0'

    # style check
    'flake8>=3.7.8',
    'isort>=4.3.4,<5.0',
    'tox>=3.14.0',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # docs
    'Sphinx==1.8.5',
    'twine==1.14.0',
]

setup(
    author="Furui Cheng",
    author_email='fr.cheng96@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="TODO: add a project description",
    entry_points={
        'console_scripts': [
            'polyphony=polyphony.cli:main',
        ],
    },
    install_requires=install_requires,
    extras_require={
        'test': tests_requires,
        'dev': dev_requires + tests_requires,
    },
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='polyphony',
    name='polyphony',
    packages=find_packages(include=['polyphony', 'polyphony.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ChengFR/polyphony',
    version='0.1.0',
    zip_safe=False,
)
