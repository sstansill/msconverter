# MeasurementSet to xarray

This is a prototype converter from MeasurementSet v2.0 to an xarray Dataset.

## Installation

The package can be loaded as a pip package. To install:

1) Clone the project
2) `cd msconverter`
3) Create a conda environment with `conda create -n msconverter python=3.11`
4) Activate the conda environment `conda activate msconverter`
5) Install pip dependencies `pip install -e .`
6) Inside a Python shell, use the converter with `from msconverter import convert`
7) `convert.convert(infile, outfile, compress=True)`


Compression slows down the conversion but results in much smaller zarr stores
