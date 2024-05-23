[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg?style=for-the-badge)](https://www.repostatus.org/#active) [![License: Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg)](https://github.com/biasvariancelabs/aitlas/blob/master/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)




# ADAF - Automatic Detection of Archaeological Features


A user-friendly software for the automatic detection of archaeological features from ALS data using convolutional neural
networks. The underlying ML models were trained on an extensive archive of ALS datasets in Ireland, labelled by experts
with three types of archaeological features (barrows, ringforts, enclosures). The core components of the tool are the
Relief Visualisation Toolbox (RVT) for processing the input data and the Artificial Intelligence Toolbox for Earth
Observation (AiTLAS), which provides access to the ML models.

<img src="adaf/media/ringfort.jpg" alt="drawing" width="200"/>
<img src="adaf/media/barrows.jpg" alt="drawing" width="200"/>
<img src="adaf/media/enclosure.jpg" alt="drawing" width="200"/>


# Installation


The installation is currently only supported on Windows 64-bit machines. The application is compatible with machines 
equipped with CUDA-enabled graphics cards, but will also work on a standard CPU where GPU processing is not possible.
We recommend creating a virtual environment with `Anaconda` and installing the requirements with `pip`.

Requirements:
* Clone the repository to your local drive
* Download the machine learning models from [Dropbox](https://www.dropbox.com/t/QRVtxUVTPRVSnYKK)
* Python 3.8 (recommended to use [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) 
or [Miniconda](https://docs.anaconda.com/free/miniconda/))


# Setting up ADAF

1. Run Anaconda Prompt (press `Windows` key and type “anaconda prompt”).

2. In the Anaconda Prompt, navigate to the installation folder by running command:
    
   ```bash
    cd <path-to-repository>\installation
    ```
   
    >`<path-to-repository>` is the location where you have downloaded and unzipped the installation files, 
   > for example `C:\temp\adaf\`

3. Create and activate a conda environment called `adaf`. Run commands:

    ```bash
    conda create -n adaf python=3.8
    conda activate adaf
    ```

    **Skip step 4 if you don’t have a CUDA enabled device!**

4. Install CUDA version of PyTorch

    > ONLY FOR CUDA COMPLIANT GPUs. When installing on a PC which has a CUDA enabled graphics card (check
   > [here](https://developer.nvidia.com/cuda-gpus) for
   > NVIDIA compliant cards) the GPU can be used to reduce processing times. If your card is compliant (also requires
   > installation of CUDA software that is not covered in this manual) install the compatible 
   >[PyTorch version](https://developer.nvidia.com/cuda-gpus).

5. Install the packages using pip:

    ```bash
    pip install GDAL-3.4.3-cp38-cp38-win_amd64.whl
    pip install aitlas-0.0.1-py3-none-any.whl
    ```
   
6. Enable the use of the AiTLAS virtual environment in Jupyter notebooks by running:

    ```bash   
    python -m ipykernel install --name aitlas
    ```

7. Close the Anaconda Prompt window. The installation is now complete.











---

**Note:** You will have to download the datasets from their respective source. You can find a link for each dataset in the respective dataset class in `aitlas/datasets/` or use the **AiTLAS Semantic Data Catalog**














