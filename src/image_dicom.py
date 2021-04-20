import numpy as np
import pydicom

import os
from typing import List, Tuple, Dict


def load_dicom(filename, normalize_scalars=False) -> Tuple[np.array, Dict]:
    """ Load volume stored in DICOM format. """
    volume = None
    header = {}

    dirname = os.path.dirname(filename)
    slices = []
    for filename in os.listdir(dirname):
        slice_ = pydicom.dcmread(os.path.join(dirname, filename))
        # Cannot use SliceLocation, since that attribute is not always present...
        if hasattr(slice_, "ImagePositionPatient"):
            slices.append(slice_)
    slices = sorted(slices, key=lambda s: s.ImagePositionPatient[2])

    if len(slices) > 1:
        dim_x = slices[0].Columns
        dim_y = slices[0].Rows
        spacing_x = slices[0].PixelSpacing[0]
        spacing_y = slices[0].PixelSpacing[1]
        # Cannot use SliceThickness either, since that is not the actual slice spacing...
        spacing_z = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
        header["dimensions"] = (dim_x, dim_y, len(slices))
        header["origin"] = (0, 0, 0)
        header["spacing"] = (spacing_x, spacing_y, spacing_z)
        dim = header["dimensions"][::-1]  # OBS! Shape should be (D,H,W)
        volume = np.zeros(dim, dtype=np.int16)
        for i in range(0, len(slices)):
            volume[i,:,:] = slices[i].pixel_array
            volume[i,:,:] += np.int16(slices[i].RescaleIntercept)
    else:
        assert(0)
    return volume, header
