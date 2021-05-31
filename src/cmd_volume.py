import gfx_utils

import numpy as np


class UpdateVolumeCmd:
    def __init__(self, volume_, subimage_, offset_, texture_=0):
        self.volume = volume_
        self.subimage = subimage_
        self.offset = offset_
        self.texture = texture_
        self._prev_subimage = None

    def apply(self):
        """Apply update to volume and its texture"""
        x, y, z = self.offset
        d, h, w = self.subimage.shape
        self._prev_subimage = np.copy(self.volume[z : z + d, y : y + h, x : x + w])
        self.volume[z : z + d, y : y + h, x : x + w] = self.subimage
        if self.texture:
            gfx_utils.update_subtexture_3d(self.texture, self.subimage, self.offset)
        return self

    def undo(self):
        """Undo update to volume and its texture"""
        x, y, z = self.offset
        d, h, w = self.subimage.shape
        self.volume[z : z + d, y : y + h, x : x + w] = self._prev_subimage
        if self.texture:
            gfx_utils.update_subtexture_3d(self.texture, self._prev_subimage, self.offset)
        return self
