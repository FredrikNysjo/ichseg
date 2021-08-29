"""
.. module:: gfx_mpr
   :platform: Linux, Windows
   :synopsis: Utils for multi-planar reformatting (MPR)

.. moduleauthor:: Fredrik Nysjo
"""

import gfx_utils

import numpy as np
import glm

MPR_PLANE_X = 0
MPR_PLANE_Y = 1
MPR_PLANE_Z = 2
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
        self.last_plane = MPR_PLANE_Z

    def update_level_range(self):
        """Update MPR level range based on current preset and minmax range"""
        preset_name = MPR_PRESET_NAMES[self.level_preset]
        preset_range = MPR_PRESET_RANGES[self.level_preset]
        if preset_name == "Auto":
            self.level_range = [v for v in self.minmax_range]
        elif preset_name != "Custom":
            self.level_range = [v for v in preset_range]

    def update_minmax_range_from_volume(self, volume):
        """Calculate and update MPR minmax range from volume

        This will also update the current MPR level range
        """
        self.minmax_range[0] = np.min(volume)
        self.minmax_range[1] = np.max(volume)
        self.update_level_range()

    def scroll_by_ray(self, volume, ray_dir, steps):
        """Scroll MPR a number of positive or negative steps along axis determined
        by the largest absolute component of the ray direction
        """
        for axis in range(0, 3):
            if abs(ray_dir[axis]) == max([abs(v) for v in ray_dir]):
                steps_new = steps * np.sign(ray_dir[axis])
                self.scroll_by_axis(volume, axis, steps_new)

    def scroll_by_axis(self, volume, axis, steps):
        """Scroll MPR a number of positive or negative steps along axis specified
        by index in range 0-2
        """
        assert axis >= 0 and axis <= 2
        delta = steps / float(volume.shape[2 - axis])
        self.planes[axis] = max(-0.4999, min(0.4999, self.planes[axis] + delta))

    def get_snapped_planes(self, volume):
        """Return copy of MPR planes snapped to voxel centers"""
        x = (self.planes[0] + 0.5) * volume.shape[2]
        y = (self.planes[1] + 0.5) * volume.shape[1]
        z = (self.planes[2] + 0.5) * volume.shape[0]
        x = (np.floor(x) + 0.5) / volume.shape[2] - 0.5
        y = (np.floor(y) + 0.5) / volume.shape[1] - 0.5
        z = (np.floor(z) + 0.5) / volume.shape[0] - 0.5
        return [x, y, z]

    def get_level_range_scaled(self, volume):
        """Return copy of level range scaled and shifted to take normalized
        texture formats into account, based on the volume's scalar type
        """
        scale, shift = (1.0, 0.0)
        if volume.dtype == np.uint8:
            scale, shift = (1.0 / 255.0, 0.0)  # Scale to range [0,1]
        elif volume.dtype == np.int16:
            scale, shift = (1.0 / 32767.0, 0.0)  # Scale to range [-1,1]
        elif volume.dtype == np.uint16:
            scale, shift = (1.0 / 65535.0, 0.0)  # Scale to range [0,1]
        return [(v + shift) * scale for v in self.level_range]

    def get_depth_from_raycasting(self, x, y, w, h, volume, view_from_local, proj_from_view):
        """Get depth value from performing raycasting against MPR planes"""
        # Obtain ray origin in view space
        ndc_pos = glm.vec3(x / float(w), 1.0 - y / float(h), 0.0) * 2.0 - 1.0
        view_pos = gfx_utils.reconstruct_view_pos(ndc_pos, proj_from_view)

        # Obtain ray origin and ray direction in volume local coordinates
        local_from_view = glm.inverse(view_from_local)
        ray_origin = glm.vec3(local_from_view * glm.vec4(view_pos, 1.0))
        ray_dir = glm.vec3(local_from_view * glm.vec4(view_pos, 0.0))
        if proj_from_view[2][3] == 0.0:  # Check if projection is orthographic
            ray_dir = -glm.vec3(local_from_view[2])
        ray_dir = glm.normalize(ray_dir)

        # Do raycasting against MPR planes
        mpr_planes = self.get_snapped_planes(volume)
        aabb_mpr = [glm.vec3(-1.0), glm.vec3(1.0)]
        hit = gfx_utils.intersect_mpr(ray_origin, ray_dir, aabb_mpr, mpr_planes)

        depth = 1.0  # Set non-hit depth output to far clipping plane
        if hit:  # Calculate actual depth output from MPR hit point
            local_pos = ray_origin + hit[0] * ray_dir
            clip_pos = proj_from_view * view_from_local * glm.vec4(local_pos, 1.0)
            depth = (clip_pos.z / clip_pos.w) * 0.5 + 0.5
        return depth
