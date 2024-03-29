"""
.. module:: image_utils
   :platform: Linux, Windows
   :synopsis: I/O utils for image data (2D images and VTK volumes)

.. moduleauthor:: Fredrik Nysjo
"""

import numpy as np
import PilLite.Image


def create_dummy_volume():
    """Create an empty dummy volume (and header) in right format"""
    volume = np.zeros([1, 1, 1], dtype=np.int16)
    header = {"dimensions": (1, 1, 1), "spacing": (1, 1, 1)}
    return volume, header


def load_image(filename):
    """Load 2D image stored in common format (png, jpg, etc.) on disk"""
    image = PilLite.Image.open(filename)
    return np.array(image)


def load_vtk(filename):
    """Load volume stored in legacy VTK format on disk. Currently only
    supports scalar data stored in binary format (not ASCII).
    """
    volume, header = None, {}
    with open(filename, "rb") as stream:
        line = stream.readline()
        while line != None:
            strings = line.decode(errors="ignore").split(" ")
            if strings[0] == "DIMENSIONS":
                header["dimensions"] = [int(s) for s in strings[1:4]]
            if strings[0] == "ORIGIN":
                header["origin"] = [float(s) for s in strings[1:4]]
            if strings[0] == "SPACING":
                header["spacing"] = [float(s) for s in strings[1:4]]
            if strings[0] == "POINT_DATA":
                header["num_points"] = int(strings[1])
            if strings[0] == "SCALARS":
                header["format"] = strings[2].strip()
            if strings[0] == "LOOKUP_TABLE" or strings[0] == "COLOR_SCALARS":
                dim = header["dimensions"][::-1]

                # Handle volumes exported with vtkStructuredPointsWriter that
                # automatically interprets unsigned char scalars as color data
                if strings[0] == "COLOR_SCALARS":
                    header["format"] = "unsigned_char"
                    ncomponents = int(strings[2])
                    assert ncomponents == 1, "Non-grayscale volumes not supported"

                if header["format"] == "unsigned_char":
                    nbytes = header["num_points"] * 1
                    volume = np.frombuffer(stream.read(nbytes), dtype=np.uint8).reshape(dim)
                    volume = volume.astype(dtype=np.uint8)  # Make writeable copy
                elif header["format"] == "short":
                    nbytes = header["num_points"] * 2
                    dt = np.dtype(np.int16).newbyteorder(">")
                    volume = np.frombuffer(stream.read(nbytes), dtype=dt).reshape(dim)
                    volume = volume.astype(dtype=np.int16)  # Reorder bytes
                elif header["format"] == "unsigned_short":
                    nbytes = header["num_points"] * 2
                    dt = np.dtype(np.uint16).newbyteorder(">")
                    volume = np.frombuffer(stream.read(nbytes), dtype=dt).reshape(dim)
                    volume = volume.astype(dtype=np.uint16)  # Reorder bytes
                elif header["format"] == "float":
                    nbytes = header["num_points"] * 4
                    dt = np.dtype(np.float32).newbyteorder(">")
                    volume = np.frombuffer(stream.read(nbytes), dtype=dt).reshape(dim)
                    volume = volume.astype(dtype=np.float32)  # Reorder bytes
                else:
                    assert False, "Scalar type not supported: " + header["format"]
                break
            line = stream.readline()
    return volume, header


def save_vtk(filename, volume, header):
    """Save volume to be stored in legacy VTK format on disk. Currently only
    supports scalar data stored in binary format (not ASCII).
    """
    with open(filename, "wb") as stream:
        w, h, d = header["dimensions"]
        sx, sy, sz = header["spacing"]
        stream.write(b"# vtk DataFile Version 3.0\n")
        stream.write(b"VTK File\nBINARY\nDATASET STRUCTURED_POINTS\n")
        stream.write(b"DIMENSIONS %d %d %d\n" % (w, h, d))
        stream.write(b"SPACING %f %f %f\n" % (sx, sy, sz))
        stream.write(b"ORIGIN 0 0 0\n")
        stream.write(b"POINT_DATA %d\n" % (w * h * d))
        stream.write(b"SCALARS scalars %s 1\n" % header["format"].encode("ascii"))
        stream.write(b"LOOKUP_TABLE default\n")
        if header["format"] == "unsigned_char":
            assert volume.dtype == np.uint8
            stream.write(volume)
        elif header["format"] == "short":
            assert volume.dtype == np.int16
            dt = np.dtype(np.int16).newbyteorder(">")
            stream.write(volume.astype(dtype=dt))
        elif header["format"] == "unsigned_short":
            assert volume.dtype == np.uint16
            dt = np.dtype(np.uint16).newbyteorder(">")
            stream.write(volume.astype(dtype=dt))
        elif header["format"] == "float":
            assert volume.dtype == np.float32
            dt = np.dtype(np.float32).newbyteorder(">")
            stream.write(volume.astype(dtype=dt))
        else:
            assert False, "Scalar type not supported: " + header["format"]
