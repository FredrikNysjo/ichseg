import numpy as np


MPR_PRESET_NAMES = ["Auto", "Custom", "Brain (CT)", "Head-neck (CT)", "Probability (8-bit)"]
MPR_PRESET_RANGES = [None, None, [0, 120], [-1024, 3071], [0, 255]]


class MPR:
    def __init__(self):
        self.planes = [0.0, 0.0, 0.0]
        self.level_preset = MPR_PRESET_NAMES.index("Auto")
        self.level_range = [0, 120]
        self.minmax_range = [-1024, 3071]
        self.show_voxels = False
        self.enabled = True
        self.scrolling = False


def mpr_update_level_range(mpr):
    """Update MPR level range based on current preset and minmax range"""
    preset_name = MPR_PRESET_NAMES[mpr.level_preset]
    preset_range = MPR_PRESET_RANGES[mpr.level_preset]
    if preset_name == "Auto":
        mpr.level_range = [v for v in mpr.minmax_range]
    elif preset_name != "Custom":
        mpr.level_range = [v for v in preset_range]


def snap_mpr_to_grid(volume, mpr_planes):
    """Snap MPR planes to voxel centers"""
    x = (mpr_planes[0] + 0.5) * volume.shape[2]
    y = (mpr_planes[1] + 0.5) * volume.shape[1]
    z = (mpr_planes[2] + 0.5) * volume.shape[0]
    x = (np.floor(x) + 0.5) / volume.shape[2] - 0.5
    y = (np.floor(y) + 0.5) / volume.shape[1] - 0.5
    z = (np.floor(z) + 0.5) / volume.shape[0] - 0.5
    return [x, y, z]
