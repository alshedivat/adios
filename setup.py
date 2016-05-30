import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='adios',
    version='0.1',
    license='MIT',
    packages=find_packages(),
    description='A Keras-based library for deep learning in the output space.',
    long_description=read('README.md'),
    author='Maruan Al-Shedivat',
    author_email='maruan@alshedivat.com',
    url='https://github.com/alshedivat/adios',
    download_url='https://github.com/alshedivat/adios/archive/master.zip',
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=['numpy>=1.5','pyyaml','argparse','keras>=1.0','Theano'],
)
