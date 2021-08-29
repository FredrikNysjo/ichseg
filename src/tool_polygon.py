"""
.. module:: tool_polygon
   :platform: Linux, Windows
   :synopsis: Polygon tool (2D only)

.. moduleauthor:: Fredrik Nysjo
"""

from tool_common import *

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


class PolygonTool:
    def __init__(self):
        self.points = []
        self.enabled = False
        self.rasterise = False
        self.clicking = False
        self.selected = -1
        self.frame_count = 0
        self.plane = TOOL_PLANE_Z
        self.antialiasing = True

    def apply(self, image, op=TOOL_OP_ADD):
        """Apply polygon tool to 2D slice of input 3D image

        Returns: tuple (subimage, offset) if successfull, otherwise None
        """
        if len(self.points) == 0:
            return None

        # Extract image slice for current drawing plane
        d, h, w = image.shape
        if self.plane == TOOL_PLANE_Z:
            offset = (0, 0, int((self.points[2] + 0.5) * d))
            slice_ = image[offset[2], :, :].astype(dtype=np.uint8)
            axes = (0, 1, 2)
            output_shape = (1, h, w)
        elif self.plane == TOOL_PLANE_Y:
            offset = (0, int((self.points[1] + 0.5) * h), 0)
            slice_ = image[:, offset[1], :].astype(dtype=np.uint8)
            axes = (0, 2, 1)
            output_shape = (d, 1, w)
        elif self.plane == TOOL_PLANE_X:
            offset = (int((self.points[0] + 0.5) * w), 0, 0)
            slice_ = image[:, :, offset[0]].astype(dtype=np.uint8)
            axes = (1, 2, 0)
            output_shape = (d, h, 1)
        else:
            assert False, "Invalid MPR plane index"

        # Construct closed 2D polygon from drawn 3D points
        npoints = len(self.points) // 3
        polygon = np.zeros((npoints + 1, 2))
        for i in range(0, npoints):
            polygon[i, 0] = self.points[3 * i + axes[0]] + 0.5
            polygon[i, 1] = self.points[3 * i + axes[1]] + 0.5
        polygon[npoints, 0] = self.points[axes[0]] + 0.5
        polygon[npoints, 1] = self.points[axes[1]] + 0.5

        # Rasterise 2D polygon into image of same size as slice
        if self.antialiasing:
            subimage = _rasterise_polygon_2d_aa(polygon, slice_)
        else:
            subimage = _rasterise_polygon_2d(polygon, slice_)

        # Combine with previous segmentation mask from slice
        if op == TOOL_OP_ADD:
            subimage = np.maximum(subimage, slice_)
        else:
            subimage = np.minimum(255 - subimage, slice_)

        # Note: returned subimage must be a volume
        subimage = subimage.reshape(output_shape)
        return subimage, offset

    def find_closest(self, p, radius=0.005):
        """Find closest polygon vertex to p (within some search radius)

        Returns: offset to closest vertex if found, otherwise -1
        """
        closest = -1
        mindist = 9999.0
        for i in range(0, len(self.points), 3):
            q = self.points[i : i + 3]
            dist = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5
            if dist < mindist and dist < radius:
                mindist = dist
                closest = i
        return closest


def _rasterise_polygon_2d(polygon, image):
    """Rasterise closed 2D polygon using fast XOR-based method"""
    h, w = image.shape
    xmin = int(np.floor(np.min(polygon[:, 0]) * w))
    xmax = int(np.ceil(np.max(polygon[:, 0]) * w))
    ymin = int(np.floor(np.min(polygon[:, 1]) * h))
    ymax = int(np.ceil(np.max(polygon[:, 1]) * h))
    output = np.zeros((h, w), dtype=np.uint8)
    nvertices = polygon.shape[0]
    for y in range(ymin, ymax):
        for i in range(0, nvertices - 1):
            v0 = polygon[i + 0, :] * (w, h)
            v1 = polygon[i + 1, :] * (w, h)
            if v1[0] < v0[0]:
                v0, v1 = v1, v0  # Swap vertices s.t. v0.x <= v1.x
            delta = v1 - v0
            t = ((y + 0.5) - v0[1]) / delta[1] if delta[1] else 0.0
            if t > 0.0 and t < 1.0:
                x = v0[0] + t * delta[0]
                output[int(y + 0.5), 0 : min(int(x + 0.5), w)] ^= 255
                output[int(y + 0.5), min(int(x + 0.5), w) : w] ^= 0
    return output


def _rasterise_polygon_2d_aa(polygon, image):
    """Rasterise closed 2D polygon using fast XOR-based method

    This version of the function computes an anti-aliased result
    by taking multiple samples when rasterising the polygon
    """
    h, w = image.shape
    accum = np.zeros(image.shape, dtype=np.uint16)
    nsamples = 0
    for y in range(0, 2):
        for x in range(0, 2):
            polygon_copy = polygon.astype(np.float32)
            polygon_copy[:, 0] += (x * 0.5 - 0.25) / w
            polygon_copy[:, 1] += (y * 0.5 - 0.25) / h
            accum += _rasterise_polygon_2d(polygon_copy, image)
            nsamples += 1
    return np.minimum(255, accum / nsamples).astype(np.uint8)
