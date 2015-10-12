# coding: utf-8

from __future__ import unicode_literals

from setuptools import setup, find_packages


setup(name="hrv",
      version="0.1.0dev",
      packages=find_packages(),
      install_requires=['scipy', 'numpy'],

      # metadata for upload to PyPI
      author="Rhenan Bartels",
      author_email="rhenan.bartels@gmail.com",
      description="Module for Heart Rate Variability analysis",
      license="PSF",
      keywords=["signal", "spectrum"],
      url="http://github.com/rhenanbartels/hrv",
)
