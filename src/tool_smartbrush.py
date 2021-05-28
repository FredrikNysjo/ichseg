from tool_common import *

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


class SmartBrushTool:
    def __init__(self):
        self.position = glm.vec4(0.0)
        self.size = 30
        self.sensitivity = 3.0
        self.delta_scaling = 2.0
        self.enabled = False
        self.painting = False
        self.momentum = 0
        self.xy = (0, 0)
        self.frame_count = 0
        self.mode = TOOL_MODE_3D
        self.plane = TOOL_PLANE_Z

    def apply(self, image, volume, texcoord, spacing, level_range=None, op=TOOL_OP_ADD):
        """Apply smart brush to input 3D image

        This tool uses the SmartPaint method from [Malmberg et al. 2014].
        Reference: F. Malmberg et al., "SmartPaint: a tool for interactive
        segmentation of medical volume images", CMBBE, 2014.

        Returns: tuple (subimage, offset) if successfull, otherwise None
        """
        if max(abs(texcoord - 0.5)) > 0.5:
            return None
        return SmartBrushTool._apply(self, image, volume, texcoord, spacing, level_range, op)

    @staticmethod
    def _apply(tool, image, volume, texcoord, spacing, level_range, op):
        """Apply smart brush to input 3D image (helper function)

        This tool uses the SmartPaint method from [Malmberg et al. 2014].
        Reference: F. Malmberg et al., "SmartPaint: a tool for interactive
        segmentation of medical volume images", CMBBE, 2014.
        """
        d, h, w = image.shape[0:3]
        center = texcoord * glm.vec3(w, h, d)
        radius = glm.vec3(tool.size)
        if tool.mode == TOOL_MODE_2D:
            radius[tool.plane] = 1  # Restrict brush to 2D plane
        lower = center - glm.vec3(radius * (spacing.x / spacing)) * 0.5 + 0.5
        upper = center + glm.vec3(radius * (spacing.x / spacing)) * 0.5 + 0.5
        lower = glm.ivec3(glm.clamp(lower, glm.vec3(0.0), glm.vec3(w, h, d)))
        upper = glm.ivec3(glm.clamp(upper, glm.vec3(0.0), glm.vec3(w, h, d)))

        xx = np.arange(lower.x - center.x + 0.5, upper.x - center.x + 0.5, 1.0)
        yy = np.arange(lower.y - center.y + 0.5, upper.y - center.y + 0.5, 1.0)
        zz = np.arange(lower.z - center.z + 0.5, upper.z - center.z + 0.5, 1.0)
        z, x, y = np.meshgrid(zz, yy, xx, indexing="ij")
        z *= spacing.z / spacing.x
        # Need to take slice spacing into account

        shift = level_range[0]
        scale = 1.0 / (level_range[1] - level_range[0])
        subimage = image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x].astype(
            dtype=np.float32
        )
        midpoint = volume[int(center.z), int(center.y), int(center.x)]
        midpoint = (midpoint - shift) * scale
        intensity = volume[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x].astype(
            dtype=np.float32
        )
        intensity -= shift
        intensity *= scale

        value = subimage * (1.0 / 255.0)
        sigma = np.maximum(0.0, 1.0 - (x ** 2 + y ** 2 + z ** 2) ** 0.5 / (tool.size * 0.5))
        delta = np.abs(intensity - midpoint) * tool.delta_scaling
        rho = np.maximum(0.0, 1.0 - delta) ** tool.sensitivity
        alpha = sigma * rho
        lambda_ = 1.0 if op == TOOL_OP_ADD else 0.0
        subimage[:, :, :] = (0.05 * alpha * lambda_ + (1.0 - 0.05 * alpha) * value) * 255.0

        subimage = np.maximum(0.0, np.minimum(255.0, subimage)).astype(dtype=np.uint8)
        image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x] = subimage
        return subimage, lower
