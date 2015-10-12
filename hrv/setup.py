from setuptools import setup, find_packages
setup(
    name = "hrv",
    version = "0.0.1",
    packages = find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['numpy>=1.8.0rc1', 'scipy>=0.13.0b1'],

    # metadata for upload to PyPI
    author = "Rhenan Bartels",
    author_email = "rhenan.bartels@gmail.com",
    description = "Module for Heart Rate Variability analysis",
    license = "PSF",
    keywords = ["signal", "spectrum"],
    url = "http://github.com/rhenanbartels/hrv",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
