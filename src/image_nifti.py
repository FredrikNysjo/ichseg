"""
.. module:: image_nifti
   :platform: Linux, Windows
   :synopsis: I/O utils for volume data in NIfTI format

.. moduleauthor:: Fredrik Nysjo
"""

import numpy as np

import struct
import gzip


def load_nii(filename):
    """Load volume stored in NIfTI format on disk. Currently only
    supports single-file .nii and .nii.gz files, and the following
    scalar type formats: uint8; int16/uint16; and float32.
    """
    volume, header = None, {}
    nii_open = gzip.open if ".gz" in filename else open
    with nii_open(filename, "rb") as stream:
        # Read header part (relevant fields only)
        hdr = stream.read(348)  # Header is always 348 bytes
        hdr_dim = struct.unpack_from("hhhhhhhh", hdr, 40)
        (hdr_datatype,) = struct.unpack_from("h", hdr, 70)
        (hdr_bitpix,) = struct.unpack_from("h", hdr, 72)
        hdr_pixdim = struct.unpack_from("ffffffff", hdr, 76)
        (hdr_vox_offset,) = struct.unpack_from("f", hdr, 108)

        # Convert to internal header format
        header["dimensions"] = [int(v) for v in hdr_dim[1:4]]
        header["origin"] = [0.0, 0.0, 0.0]  # FIXME
        header["spacing"] = [float(v) for v in hdr_pixdim[1:4]]
        header["transform"] = np.identity(3)
        header["num_points"] = hdr_dim[1] * hdr_dim[2] * hdr_dim[3]

        # Read volume part (voxel byte data)
        nbytes = hdr_dim[1] * hdr_dim[2] * hdr_dim[3] * (hdr_bitpix // 8)
        _ = stream.read(int(hdr_vox_offset) - 348)
        img = stream.read(nbytes)

        # Convert to volume in NumPy array format
        dim = header["dimensions"][::-1]
        if hdr_datatype == 2:  # code for unsigned char
            header["format"] = "unsigned_char"
            volume = np.frombuffer(img, dtype=np.uint8).reshape(dim)
            volume = volume.astype(dtype=np.uint8)  # Make writeable copy
        elif hdr_datatype == 4:  # code for signed short
            header["format"] = "short"
            volume = np.frombuffer(img, dtype=np.int16).reshape(dim)
            volume = volume.astype(dtype=np.int16)  # Make writeable copy
        elif hdr_datatype == 512:  # code for unsigned short
            header["format"] = "unsigned_short"
            volume = np.frombuffer(img, dtype=np.uint16).reshape(dim)
            volume = volume.astype(dtype=np.uint16)  # Make writeable copy
        elif hdr_datatype == 16:  # code for float (32-bit)
            header["format"] = "float"
            volume = np.frombuffer(img, dtype=np.float32).reshape(dim)
            volume = volume.astype(dtype=np.float32)  # Make writeable copy
        elif hdr_datatype == 64:  # code for double (64-bit)
            # Convert scalars to floats so that we do not have to deal with
            # double-precision when doing rendering or computations
            header["format"] = "float"
            volume = np.frombuffer(img, dtype=np.float64).reshape(dim)
            volume = volume.astype(dtype=np.float32)  # Make writeable copy
        else:
            assert False, "Scalar type not supported: Unknown NIfTI datatype: " + hdr_datatype
    return volume, header


def save_nii(filename, volume, header):
    """Save volume to be stored in NIfTI format on disk. Currently only
    supports single-file .nii and .nii.gz files, and the following
    scalar type formats: uint8; int16/uint16; and float32.
    """
    nii_open = gzip.open if ".gz" in filename else open
    with nii_open(filename, "wb") as stream:
        # Convert internal header format to NIfTI fields
        hdr_dim = [3, *header["dimensions"], 0, 0, 0, 0]
        hdr_pixdim = [1, *header["spacing"], 0, 0, 0, 0]
        hdr_vox_offset = 352.0  # OBS! Has to be a float!

        # Find which NIfTI scalar format we should use
        if header["format"] == "unsigned_char":
            assert volume.dtype == np.uint8
            hdr_datatype, hdr_bitpix = 2, 8
        elif header["format"] == "short":
            assert volume.dtype == np.int16
            hdr_datatype, hdr_bitpix = 4, 16
        elif header["format"] == "unsigned_short":
            assert volume.dtype == np.uint16
            hdr_datatype, hdr_bitpix = 512, 16
        elif header["format"] == "float":
            assert volume.dtype == np.float32
            hdr_datatype, hdr_bitpix = 16, 32
        else:
            assert False, "Scalar type not supported: " + header["format"]

        # Write header part (relevant fields only)
        hdr = bytearray(348)  # Header is always 348 bytes
        struct.pack_into("i", hdr, 0, 348)
        struct.pack_into("hhhhhhhh", hdr, 40, *hdr_dim)
        struct.pack_into("h", hdr, 70, hdr_datatype)
        struct.pack_into("h", hdr, 72, hdr_bitpix)
        struct.pack_into("ffffffff", hdr, 76, *hdr_pixdim)
        struct.pack_into("f", hdr, 108, hdr_vox_offset)
        struct.pack_into("i", hdr, 344, 0x6E2B3100)  # Magic string
        stream.write(hdr + b"\0\0\0\0")  # Pad to match vox_offset

        # Write volume part (voxel byte data)
        stream.write(volume)
