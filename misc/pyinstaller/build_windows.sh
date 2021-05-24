#!/bin/bash

pyinstaller.exe --noconfirm --onedir --console --runtime-hook=./misc/pyinstaller/hook.py src/main.py &&
mv dist/main/main.exe main-cli.exe &&
pyinstaller.exe --noconfirm --onedir --windowed --runtime-hook=./misc/pyinstaller/hook.py src/main.py &&
mv main-cli.exe dist/main/ &&

mkdir dist/main/numpy/.libs &&
mkdir dist/main/scipy/.libs &&
mv dist/main/*.gfortran-win_amd64.dll dist/main/scipy/.libs/ &&
mv dist/main/scipy/.libs/libopenblas.GK7* dist/main/numpy/.libs/ &&

mkdir dist/main/lib &&
mv dist/main/*.pyd dist/main/lib/ &&
mv dist/main/*.dll dist/main/lib/ &&

mv dist/main/lib/python38.dll dist/main/ &&
cp misc/pyinstaller/glfw3.dll dist/main/
