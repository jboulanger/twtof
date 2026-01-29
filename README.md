# Reader for Tofwerk time of flight mass spectroscopy data

This module interprets Tofwerk time of flight mass spectroscopy data HDF5 file as Python objects and xarray DataArrays.

## Installation
To install the module only:
```bash
pip install git+https://github.com/jboulanger/twtof.git
```

To install the module, create an environment and use the notebooks:
```bash
git clone https://github.com/jboulanger/twtof.git
cd twotof
conda env create -f environment.yml
conda activate tof
pip install -e .
```

## Usage

Several jupyter notebooks are provided:
- `Example.ipynb` : Basic loading of the file
- `Export_images.ipynb`: Export images as TIF to be loaded in other software
- `Export_ROIs,ipynb` : Create ROIs and measure mean spectra in each ROIs
- `Correlative_SIMS-EM.ipynb`: Alignment of SIMS and EM data


