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

## TODO

- [ ] Add features list
