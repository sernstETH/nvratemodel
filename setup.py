from setuptools import setup

setup(name='nvratemodel',
      version='1.0',
      description='Rate models to simulate the photo-physics of the Nitrogen-Vacancy center in diamond.',
      author='Stefan Ernst',
      author_email='sternst@ethz.ch',
      license='all rights reserved',
      packages=find_packages(),
      install_requires=['numpy','scipy','matplotlib','numba'],
      zip_safe=False)
