from tool_common import *

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


class BrushTool:
    def __init__(self):
        self.position = glm.vec4(0.0)
        self.size = 30
        self.enabled = False
        self.painting = False
        self.frame_count = 0
        self.mode = TOOL_MODE_3D
        self.plane = TOOL_PLANE_Z
        self.antialiasing = True

    def apply(self, image, texcoord, spacing, op=TOOL_OP_ADD):
        """Apply brush tool to input 3D image

        Returns: tuple (subimage, offset) if successfull, otherwise None
        """
        if max(abs(texcoord - 0.5)) > 0.5:
            return None
        return BrushTool._apply(self, image, texcoord, spacing, op)

    @staticmethod
    def _apply(tool, image, texcoord, spacing, op):
        """Apply brush tool to input 3D image (helper function)"""
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
        subimage = (tool.size * 0.5 - (x ** 2 + y ** 2 + z ** 2) ** 0.5 + 0.5) * 255.99
        if tool.antialiasing == False:
            # Apply thresholding to create binary mask
            subimage = (subimage >= 127.5).astype(dtype=np.uint8) * 255
        subimage = np.maximum(0, np.minimum(255, subimage)).astype(dtype=np.uint8)

        if op == TOOL_OP_ADD:
            image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x] = np.maximum(
                image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x], subimage
            )
        else:
            image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x] = np.minimum(
                image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x], 255 - subimage
            )
        subimage = image[lower.z : upper.z, lower.y : upper.y, lower.x : upper.x]
        return subimage, lower
