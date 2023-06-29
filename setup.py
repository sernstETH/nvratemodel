from setuptools import setup, find_packages

setup(name='nvratemodel',
      version='1.0',
      description='Numerical rate models to simulate the photo-physics of the Nitrogen-Vacancy (NV) center in diamond.',
      author='Stefan Ernst',
      author_email='sternst@ethz.ch',
      license='all rights reserved',
      packages=find_packages(),
      install_requires=['numpy','scipy','matplotlib','numba'],
      zip_safe=False)
