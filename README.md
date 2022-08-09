# ichseg

An interactive tool and GUI for segmenting and annotating volumes stored in DICOM or VTK-format

## Example screenshots

![Screenshot](screenshot.png?raw=true "Screenshot")

## General requirements

- Hardware: A GPU supporting at least OpenGL 4.1
- OS: Windows 10 or Ubuntu 18.04+ (not tested on macOS)

## Python installation (via Anaconda and pip):

1. Install the Anaconda (Miniconda) package manager for Python 3.8 from [here](https://docs.conda.io/en/latest/miniconda.html). On Linux, you can also install it like this:
```
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
    sh Miniconda3-latest-Linux-x86_64.sh
```
2. Create a new virtual environment (ichseg) for the application, from the terminal or Anaconda command line:
```
    conda create --name ichseg python=3.8
```
3. Activate the virtual environment and install the required Python dependendecies (via pip):
```
    conda activate ichseg
    pip install -r requirements.txt
```

## Basic usage

To run the GUI in the virtual environment from the terminal or Anaconda command line:

    conda activate ichseg
    python src/main.py

## Other notes

### Code style

This project uses the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html), with the exception of line length, which is set to 100 instead of the default 88 characters. For automatic formatting, the [black code formatter](https://pypi.org/project/black/) can be installed via pip,

    pip install black

and then applied to a source file like this:

    black --line-length=100 sourcefile.py

## License

The code is provided under the MIT license (see LICENSE.md).

## Funding

This work was supported by Vinnova grant (2020-03616) "Computer-aided Glioblastoma and Intracranial Aneurysm Treatment Response Quantification in Neuroradiology", and by AIDA technical fellowship project "Decision Support Tool for the Detection of Acute Intracranial Hemorrhage on Non-Contrast CT-Method Development".
