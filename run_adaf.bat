@echo off 
rem  Get batch script parent folder
for %%i in ("%~dp0.") do set "ADAF_DIR=%%~fi"
cd %ADAF_DIR%

rem Get path to aitlas conda environment
for /f "delims=" %%i in ('where conda 2^>nul') do (
    set "CONDA_DIR=%%i"
)
for %%I in ("%CONDA_DIR%\..\..") do set "CONDA_DIR=%%~fI"

:: enable anaconda environment
call %CONDA_DIR%\Scripts\activate.bat aitlas

python -m notebook ADAF_main.ipynb

exit
