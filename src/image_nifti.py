import numpy as np

import struct
import gzip


def load_nii(filename):
    """Load volume stored in NIfTI format on disk. Currently only
    supports single-file .nii files.
    """
    volume, header = None, {}
    nii_open = gzip.open if ".gz" in filename else open
    with nii_open(filename, "rb") as stream:
        # Read header part (relevant fields only)
        hdr = stream.read(348)
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
        else:
            assert False, "Scalar type not supported: Unknown NIfTI datatype: " + hdr_datatype
    return volume, header
