# TRAIL 2025 Workshop: Automatic detection of archaeological features


## Software and data requirements
1. Please download all the required data from here:
[Workshop data download](https://oblak.zrc-sazu.si/index.php/s/82WWxTrfpmtwEZY/download)
    > **Bring your own data: you are encouraged to bring your own archaeological DEMs (e.g. GeoTIFF, VRT), preferably of areas containing barrows or other circular archaeological objects.**

2. The workshop will require you to have the following software packages installed:
  - QGIS one of the latest LTR versions (preferably 3.40): [QGIS download](https://qgis.org/download/)
    - QGIS LandTalk.AI plugin. Check the requirements for the sign in (ChatGPT, Gemini).
    - QGIS RVT plugin.
  - Miniconda, required for installation of ADAF [Miniconda download](https://www.anaconda.com/docs/getting-started/miniconda/install)
  - Files with trained Machine learning models [ML models download](https://doi.org/10.5281/zenodo.15848662).

## Step by step instructions for ADAF installation

1. Clone the repository to your local drive and extract the repository to your desired location
    > For example: `C:\trail_workshops\ws2_adaf`. This is no considered your `<path-to-repository>`.

3. Move the downloaded ML weights to `<path-to-repository>\adaf\ml_models`

   ML models are saved in *.TAR format. Do not extract the individual TAR files, all you need to do is move them to the correct folder inside the ADAF folder:

   `<path-to-repository>\ws2_adaf\adaf\ml_models`

5. Run Anaconda Prompt

    > To open the Anaconda Prompt you can press the `Windows` key, type “Anaconda Prompt”, and select the application from the search results.

6. In the Anaconda Prompt, navigate to the installation folder by running commands:
    
   ```bash
   cd <path-to-repository>
   ```
   
   `<path-to-repository>` is the location where you have downloaded and unzipped the installation files (for example `C:\temp\adaf\`)

7. Create and activate a conda environment called `adaf`. Run commands:

    ```bash
    conda create -n adaf python=3.9 -y
    conda activate adaf
    ```
    
8. Install required Python packages

    * GDAL and connected Python dependencies
        ```bash
        pip install "numpy==1.26.4" https://github.com/cgohlke/geospatial-wheels/releases/download/v2024.9.22/GDAL-3.9.2-cp39-cp39-win_amd64.whl https://github.com/cgohlke/geospatial-wheels/releases/download/v2024.9.22/fiona-1.10.1-cp39-cp39-win_amd64.whl https://github.com/cgohlke/geospatial-wheels/releases/download/v2024.9.22/rasterio-1.3.11-cp39-cp39-win_amd64.whl "opencv-python<4.12" "opencv-python-headless<4.12"
        ```
        > Downloads and installs the unofficial wheels for:
        > gdal 3.10.3
        > rasterio 1.4.3
        > fiona 1.10.1
        > numpy 1.26.4, opencv-python <4.12 and opencv-python-headless <4.12 are required for compatibility

    * Install PyTorch for CUDA
        > **Skip this step if you don’t have a CUDA enabled device!**

        ONLY FOR CUDA COMPLIANT GPUs. When installing on a PC which has a CUDA enabled graphics card (view list of [NVIDIA compliant cards](https://developer.nvidia.com/cuda-gpus)) the GPU can be used to reduce processing times.
  
        Run to check the CUDA version:
        ```bash
        nvidia-smi
        nvcc --version
        ```

        Install the compatible [PyTorch version](https://developer.nvidia.com/cuda-gpus).
    
   * AiTLAS
        This package installs AiTLAS
        ```bash
        pip install ./installation/aitlas-0.0.1-py3-none-any.whl
        ```
   
10. Enable the use of the `adaf` virtual environment in Jupyter notebooks by running:

    ```bash   
    python -m ipykernel install --name adaf --user
    ```

11. Run Jupyter Notebook with the following command:

    ```bash   
    jupyter lab ADAF_main.ipynb
    ```


