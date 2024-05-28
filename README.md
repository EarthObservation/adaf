[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg?style=for-the-badge)](https://www.repostatus.org/#active) [![License: Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg)](https://github.com/biasvariancelabs/aitlas/blob/master/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)




# ADAF - Automatic Detection of Archaeological Features


A user-friendly software for the automatic detection of archaeological features from ALS data using convolutional neural
networks. The underlying ML models were trained on an extensive archive of ALS datasets in Ireland, labelled by experts
with three types of archaeological features (barrows, ringforts, enclosures). The core components of the tool are the
Relief Visualisation Toolbox (RVT) for processing the input data and the Artificial Intelligence Toolbox for Earth
Observation (AiTLAS), which provides access to the ML models.

<img src="adaf/media/ringfort.jpg" alt="drawing" width="200"/> <img src="adaf/media/barrows.jpg" alt="drawing" width="200"/> <img src="adaf/media/enclosure.jpg" alt="drawing" width="200"/>


# Installation

The installation is currently only supported on Windows 64-bit machines. The application is compatible with machines 
equipped with CUDA-enabled graphics cards, but will also work on a standard CPU where GPU processing is not possible.
We recommend creating a virtual environment with `Anaconda` and installing the requirements with `pip`. 
See the **Step by step instructions** below.


## Requirements
* Python 3.8 (recommended to use conda virtual environment: download and install
[Miniconda](https://docs.anaconda.com/free/miniconda/) or
[Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/))

* Files with trained Machine learning models. Download from [Dropbox](https://www.dropbox.com/t/QRVtxUVTPRVSnYKK).

    > **Warning:** Download contians data with large file size **~5GB** in total. This includes 8 pretrained ML models saved as TAR files - 4 for semantic segmentation and 4 for object detection.


## Step by step instructions

1. Clone the repository to your local drive

2. Move the TAR files to `<path-to-repository>\adaf\ml_models`

    > Do not change the filenames and make sure that files are copied to the exact location!

3. Run Anaconda Prompt (press `Windows` key and type “anaconda prompt”).

4. In the Anaconda Prompt, navigate to the installation folder by running commands:
    
   ```bash
   cd <path-to-repository>
   cd installation
   ```
   
    >`<path-to-repository>` is the location where you have downloaded and unzipped the installation files, 
   > for example `C:\temp\adaf\`

5. Create and activate a conda environment called `adaf`. Run commands:

    ```bash
    conda create -n adaf python=3.8
    conda activate adaf
    ```
    
---
6. Install PyTorch for CUDA


    **Skip this step if you don’t have a CUDA enabled device!**

    > ONLY FOR CUDA COMPLIANT GPUs. When installing on a PC which has a CUDA enabled graphics card (check
    > [here](https://developer.nvidia.com/cuda-gpus) for
    > NVIDIA compliant cards) the GPU can be used to reduce processing times. If your card is compliant (also requires
    > installation of CUDA software that is not covered in this manual) install the compatible 
    > [PyTorch version](https://developer.nvidia.com/cuda-gpus).

---

7. Install the packages using pip:

    ```bash
    pip install GDAL-3.4.3-cp38-cp38-win_amd64.whl
    pip install aitlas-0.0.1-py3-none-any.whl
    ```
   
8. Enable the use of the AiTLAS virtual environment in Jupyter notebooks by running:

    ```bash   
    python -m ipykernel install --name adaf
    ```

9. Navigate back to main adaf folder and run Jupyter Notebook with the following command:

    ```bash   
    cd ..
    jupyter notebook ADAF_main.ipynb
    ```

