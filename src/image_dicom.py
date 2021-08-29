"""
.. module:: image_dicom
   :platform: Linux, Windows
   :synopsis: I/O utils for volume data in DICOM format

.. moduleauthor:: Fredrik Nysjo
"""

import numpy as np
import scipy as sp
import scipy.ndimage
import pydicom

import os
import time


def load_dicom(filename):
    """Load volume stored in DICOM format."""
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
        sx = float(slices[0].PixelSpacing[0])
        sy = float(slices[0].PixelSpacing[1])
        # Cannot use SliceThickness either, since that is not the actual slice spacing...
        sz = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]

        header["dimensions"] = (dim_x, dim_y, len(slices))
        header["origin"] = (0, 0, 0)
        header["spacing"] = (sx, sy, sz)
        header["transform"] = _compute_affine_transform(slices, sx, sy, sz)
        header["format"] = "short"  # Assume data is in 16-bit format
        dim = header["dimensions"][::-1]  # OBS! Shape should be (D,H,W)
        volume = np.zeros(dim, dtype=np.int16)
        for i in range(0, len(slices)):
            volume[i, :, :] = slices[i].pixel_array
            volume[i, :, :] += np.int16(slices[i].RescaleIntercept)
    else:
        assert False, "Single slice DICOM not supported"
    return volume, header


def _compute_affine_transform(slices, sx, sy, sz):
    """Compute affine transform matrix from slice metadata and estimated
    voxel spacing
    """
    orient = np.array(slices[0].ImageOrientationPatient)
    if all(orient == np.array([1, 0, 0, 0, 1, 0])):
        matrix = np.identity(3)
    else:
        pos_first = np.array(slices[0].ImagePositionPatient)
        pos_last = np.array(slices[-1].ImagePositionPatient)
        pos_diff = pos_last - pos_first
        # TODO Calculate basis vectors for affine transform
        # (Need to verify this on more datasets with gantry tilt)
        znew = pos_diff / max(1e-9, np.linalg.norm(pos_diff))
        ynew = np.array([orient[3], orient[4], -orient[5] * (sx / sz)])
        xnew = np.array([orient[0], orient[1], -orient[2] * (sx / sz)])
        # Store basis vectors in swizzled order (ZYX instead of XYZ)
        matrix = np.array([znew[::-1], ynew[::-1], xnew[::-1]])
    return matrix


def _compute_output_shape(volume, header):
    """Compute output shape for affine transform

    Right now, this just doubles the number of slices, so the affine transform
    itself is not used in the calculations
    """
    output_shape = np.array(volume.shape)
    output_shape = [int(x) for x in output_shape]
    output_shape[0] *= 2
    return tuple(output_shape)


def resample_volume(volume, header):
    """Resample input volume that has affine transform stored in header

    This also doubles the number of slices in the resampled volumed,
    and resets its affine transform to the identity matrix

    Returns: tuple (volume_output, header_output) after resampling
    """
    if "transform" not in header:
        header["transform"] = np.identity(3)

    sx, sy, sz = header["spacing"]
    matrix = np.array(header["transform"])

    output_shape = _compute_output_shape(volume, header)
    matrix[0] *= 0.5
    sz *= 0.5
    center = np.array(volume.shape) * 0.5
    center_output = np.array(output_shape) * 0.5
    offset = center - center_output.dot(matrix)

    header_output = {}
    header_output["dimensions"] = output_shape[::-1]
    header_output["origin"] = header["origin"]
    header_output["spacing"] = (sx, sy, sz)
    header_output["transform"] = np.identity(3)
    header_output["format"] = header["format"]

    print("Resampling volume...")
    tic = time.time()
    volume_output = sp.ndimage.interpolation.affine_transform(
        volume, matrix.T, offset, output_shape, order=2, cval=0.0, prefilter=False
    )
    print("Done (elapsed time: %.2f s)" % (time.time() - tic))

    return volume_output, header_output
