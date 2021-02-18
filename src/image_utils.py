import numpy as np
import PilLite.Image

from typing import List, Tuple, Dict


def load_image(filename) -> np.array:
    """ Load 2D image stored in common format (png, jpg, etc.) on disk """
    image = PilLite.Image.open(filename)
    return np.array(image)


def load_vtk(filename, normalize_scalars=False) -> Tuple[np.array, Dict]:
    """ Load volume stored in legacy VTK format on disk. Currently only
        supports scalar data stored in binary format (not ASCII).
    """
    volume = None
    header = {}
    with open(filename, 'rb') as stream:
        line = stream.readline()
        while line != None:
            strings = line.decode(errors='ignore').split(" ")
            if strings[0] == "DIMENSIONS":
                header["dimensions"] = [int(s) for s in strings[1:4]]
            if strings[0] == "ORIGIN":
                header["origin"] = [float(s) for s in strings[1:4]]
            if strings[0] == "SPACING":
                header["spacing"] = [float(s) for s in strings[1:4]]
            if strings[0] == "POINT_DATA":
                header["num_points"] = int(strings[1])
            if strings[0] == "SCALARS":
                header["format"] = strings[2]
            if strings[0] == "LOOKUP_TABLE":
                dim = header["dimensions"][::-1]
                if header["format"] == "unsigned_char":
                    volume = np.frombuffer(stream.read(), dtype=np.uint8).reshape(dim)
                elif header["format"] == "short":
                    dt = np.dtype(np.int16).newbyteorder(">")
                    volume = np.frombuffer(stream.read(), dtype=dt).reshape(dim)
                    if normalize_scalars:
                        volume = (volume + 1024) * 8
                else:
                    assert(0)
                break
            line = stream.readline()
    return volume, header


def save_vtk(filename, volume, header) -> None:
    """ Save volume to be stored in legacy VTK format on disk. Currently only
        supports scalar data stored in binary format (not ASCII).
    """
    assert(volume.dtype == np.uint8)

    with open(filename, 'wb') as stream:
        w, h, d = header["dimensions"]
        sx, sy, sz = header["spacing"]
        stream.write(b"# vtk DataFile Version 3.0\n")
        stream.write(b"VTK File\nBINARY\nDATASET STRUCTURED_POINTS\n")
        stream.write(b"DIMENSIONS %d %d %d\n" % (w, h, d))
        stream.write(b"SPACING %f %f %f\n" % (sx, sy, sz))
        stream.write(b"ORIGIN 0 0 0\n")
        stream.write(b"POINT_DATA %d\n" % (w * h * d))
        stream.write(b"SCALARS scalars unsigned_char 1\n")
        stream.write(b"LOOKUP_TABLE default\n")
        stream.write(volume);
