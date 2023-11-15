[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg?style=for-the-badge)](https://www.repostatus.org/#active) [![License: Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg)](https://github.com/biasvariancelabs/aitlas/blob/master/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)




# ADAF - Automatic Detection of Archaeological Features

todo - Description of ADAF

<img src="adaf/media/ringfort.jpg" alt="drawing" width="200"/>
<img src="adaf/media/barrows.jpg" alt="drawing" width="200"/>
<img src="adaf/media/enclosure.jpg" alt="drawing" width="200"/>

# Getting started



# Installation

The best way to install `aitlas`, is if you create a virtual environment with `Anaconda` and install the  requirements with `pip`.
Here are the steps:
- Install Anaconda from the [Anaconda official download site](https://www.anaconda.com/download) 

- Open `Anaconda Prompt` and navigate to the folder where you cloned the ADAF repository
```bash
cd d:\adaf
```

- Create a conda virtual environment with Python version 3.8
```bash
conda create -n adaf python=3.8
```
>In this example `adaf` is the name of the environment


- Use the virtual environment
```bash
conda activate adaf
```

- Before installing `aitlas` on Windows it is recommended to install the GDAL package 
from [Unofficial Windows wheels repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/):
> -	select an appropriate download version for your system, e.g. for 64-bit Windows with Python 3.8 download GDAL-3.4.3-cp38-cp38-win_amd64.whl
> - save downloaded files to the `adaf` repository folder and  install them
```bash
pip install GDAL-3.4.1-cp38-cp38-win_amd64.whl 
```

- Install the requirements
```bash
pip install -r requirements.txt
```

- Install AiTLAS as a python package
```bash
python setup.py build
python setup.py install
```

- Install jupyter notebook kernel
```bash
python -m ipykernel install --user --name=adaf
```

- Run jupyter notebooks
```bash
jupyter notebook
```
---

**Note:** You will have to download the datasets from their respective source. You can find a link for each dataset in the respective dataset class in `aitlas/datasets/` or use the **AiTLAS Semantic Data Catalog**














