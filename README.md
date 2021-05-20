# ichseg

Tool for visualizing and segmenting intracranial hemmorhages from CT volume data

## Example screenshots

![Screenshot](screenshot.png?raw=true "Screenshot")

## System requirements

- A GPU supporting at least OpenGL 4.1

## Installing dependencies on Linux

1. Install Anaconda (Miniconda) package manager:
```
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
    sh Miniconda3-latest-Linux-x86_64.sh
```
2. Activate the conda environment:
```
    conda activate
```
3. Install required Python dependencies:
```
    conda install numpy scipy numba
    pip install PyOpenGL PyOpenGL_accelerate glfw imgui[glfw] pyglm Pil-Lite   
    pip install pylibjpeg pylibjpeg-libjpeg pydicom
```
## Installing dependencies on Windows

TODO.

## Installing dependencies on macOS

TODO.

## Basic usage

TODO.

## Other notes

### Code style

This project uses the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html), with the exception of line length, which is set to 100 instead of the default 88 characters. For automatic formatting, the [black code formatter](https://pypi.org/project/black/) can be installed via pip,

    pip install black

and then applied to a source file like this:

    black --line-length=100 sourcefile.py
