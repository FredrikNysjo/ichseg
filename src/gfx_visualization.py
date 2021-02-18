import numpy as np

class MPR:
    def __init__(self):
        self.planes = [0.0, 0.0, 0.0]
        self.level_range = [0.0, 120.0]
        self.show_voxels = False
        self.enabled = True
        self.scrolling = False


def snap_mpr_to_grid(volume, mpr_planes):
    """ Snap MPR planes to voxel centers """
    x = (mpr_planes[0] + 0.5) * volume.shape[2]
    y = (mpr_planes[1] + 0.5) * volume.shape[1]
    z = (mpr_planes[2] + 0.5) * volume.shape[0]
    x = (np.floor(x) + 0.5) / volume.shape[2] - 0.5
    y = (np.floor(y) + 0.5) / volume.shape[1] - 0.5
    z = (np.floor(z) + 0.5) / volume.shape[0] - 0.5
    return [x, y, z]