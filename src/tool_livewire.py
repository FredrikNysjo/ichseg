from tool_common import *
from tool_polygon import PolygonTool

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


class LivewireTool:
    def __init__(self):
        self.graph = None
        self.pred = None
        self.dist = None
        self.path = []
        self.points = []
        self.markers = []
        self.smoothing = True
        self.enabled = False
        self.rasterise = False
        self.clicking = False
        self.plane = TOOL_PLANE_Z
        self.antialiasing = False

    def apply(self, image, op=TOOL_OP_ADD):
        """Apply livewire tool to 2D slice of input 3D image

        Returns: tuple (subimage, offset) if successfull, otherwise None
        """
        # XXX Re-use the existing code for the polygon tool, since a livewire
        # is basically just a polygon with a vertex for each pixel or voxel
        return PolygonTool.apply(self, image, op)

    def update_graph(self, image, texcoord, level_range):
        """Update livewire graph from 2D slice of input 3D image"""
        if len(self.path):
            return  # Active livewire should already have a graph

        d, h, w = image.shape
        if self.plane == TOOL_PLANE_Z:
            slice_ = image[int(texcoord.z * d), :, :].astype(np.float32)
            seed = int(texcoord.y * h) * w + int(texcoord.x * w)
        elif self.plane == TOOL_PLANE_Y:
            slice_ = image[:, int(texcoord.y * h), :].astype(np.float32)
            seed = int(texcoord.z * d) * w + int(texcoord.x * w)
        elif self.plane == TOOL_PLANE_X:
            slice_ = image[:, :, int(texcoord.x * w)].astype(np.float32)
            seed = int(texcoord.z * d) * h + int(texcoord.y * h)
        else:
            assert False, "Invalid MPR plane index"

        shift = level_range[0]
        scale = 1.0 / max(1e-9, level_range[1] - level_range[0])
        slice_normalized = np.maximum(0.0, np.minimum(1.0, (slice_ - shift) * scale))

        self.graph = _create_graph_from_image(slice_normalized)
        _update_edge_weights(self.graph, slice_normalized, 0.0, 1.0)

        self.dist, self.pred = _compute_dijkstra(self.graph, seed)
        self.path.append(seed)

    def update_path(self, image, texcoord, level_range, clicking):
        """Update livewire path from current 2D image graph"""
        d, h, w = image.shape
        if self.plane == TOOL_PLANE_Z:
            seed = int(texcoord.y * h) * w + int(texcoord.x * w)
            offset = texcoord.z - 0.5
        elif self.plane == TOOL_PLANE_Y:
            seed = int(texcoord.z * d) * w + int(texcoord.x * w)
            offset = texcoord.y - 0.5
        elif self.plane == TOOL_PLANE_X:
            seed = int(texcoord.z * d) * h + int(texcoord.y * h)
            offset = texcoord.x - 0.5
        else:
            assert False, "Invalid MPR plane index"

        path = _compute_shortest_path(self.pred, self.path[-1], seed)
        _update_livewire(self, path, offset, image)
        if self.smoothing:
            _smooth_livewire(self)

        if clicking:
            self.dist, self.pred = _compute_dijkstra(self.graph, seed)
            self.path.extend(path)
            self.path.append(seed)


def _create_graph_from_image(image):
    """Constructs a sparse matrix for a 4-connected image graph"""
    h, w = image.shape
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    offsets = (-w, -1, 1, w)
    graph = sp.sparse.diags(weights, offsets, shape=(w * h, w * h), format="csr")
    return graph


def _update_edge_weights(graph, image, alpha0, alpha1):
    """Update graph edge weights from image values"""
    assert graph.format == "csr" or graph.format == "lil"
    h, w = image.shape
    for y in range(1, h - 1):
        diff0 = image[y - 1, 1 : w - 1] - image[y, 1 : w - 1]
        diff1 = image[y + 0, 0 : w - 2] - image[y, 1 : w - 1]
        diff2 = image[y + 0, 2 : w - 0] - image[y, 1 : w - 1]
        diff3 = image[y + 1, 1 : w - 1] - image[y, 1 : w - 1]

        grad0 = image[y - 1, 0 : w - 2] + image[y, 0 : w - 2]
        grad0 -= image[y - 1, 2:w] + image[y, 2:w]
        grad1 = image[y - 1, 0 : w - 2] + image[y - 1, 1 : w - 1]
        grad1 -= image[y + 1, 0 : w - 2] + image[y + 1, 1 : w - 1]
        grad2 = image[y - 1, 1 : w - 1] + image[y - 1, 2:w]
        grad2 -= image[y + 1, 1 : w - 1] + image[y + 1, 2:w]
        grad3 = image[y + 1, 0 : w - 2] + image[y, 0 : w - 2]
        grad3 -= image[y + 1, 2:w] + image[y, 2:w]

        w0 = 1.0 / (alpha0 * np.abs(diff0) + alpha1 * np.abs(grad0) + 1e-3)
        w1 = 1.0 / (alpha0 * np.abs(diff1) + alpha1 * np.abs(grad1) + 1e-3)
        w2 = 1.0 / (alpha0 * np.abs(diff2) + alpha1 * np.abs(grad2) + 1e-3)
        w3 = 1.0 / (alpha0 * np.abs(diff3) + alpha1 * np.abs(grad3) + 1e-3)
        for x in range(1, w - 1):
            idx = y * w + x
            # Updating the matrix row data directly is much faster than
            # accessing indivdual elements
            weights = (w0[x - 1], w1[x - 1], w2[x - 1], w3[x - 1])
            if graph.format == "lil":
                graph.data[idx] = weights
            if graph.format == "csr":
                graph.data[graph.indptr[idx] : graph.indptr[idx + 1]] = weights


def _compute_dijkstra(graph, seed):
    """Computes distances to seed point(s) in the graph, and
    predecessor matrix, using Dijkstra's algorithm
    """
    dist, pred = sp.sparse.csgraph.dijkstra(
        graph, directed=False, indices=seed, return_predecessors=True
    )
    return dist, pred


def _compute_shortest_path(pred, a, b):
    """Compute shortest path (as index list) from predecessor matrix"""
    if b < 0 or b >= len(pred):
        return []  # This can happen when cursor is moved outside window
    path = []
    idx = pred[b]
    while idx != -9999 and idx != a:
        path.append(idx)
        idx = pred[idx]
    path.reverse()
    return path


def _update_livewire(livewire, path_new, offset, volume):
    """Update line segments of livewire from its current path and
    new path not yet appended to the livewire (for preview)
    """
    d, h, w = volume.shape
    points = []
    for idx in livewire.path + path_new:
        if livewire.plane == TOOL_PLANE_Z:
            x = (idx % w) / float(w) - 0.5
            y = (idx // w) / float(h) - 0.5
            z = offset
        elif livewire.plane == TOOL_PLANE_Y:
            x = (idx % w) / float(w) - 0.5
            z = (idx // w) / float(d) - 0.5
            y = offset
        elif livewire.plane == TOOL_PLANE_X:
            x = offset
            y = (idx % h) / float(h) - 0.5
            z = (idx // h) / float(d) - 0.5
        points.extend((x, y, z))
    livewire.points = points


def _smooth_livewire(livewire, iterations=5):
    """Apply smoothing to line segments in livewire"""
    points = livewire.points
    output = [x for x in points]
    for j in range(0, iterations):
        for i in range(3, len(points) - 3):
            output[i] = (points[i - 3] + points[i + 3]) * 0.25 + points[i] * 0.5
        points = [x for x in output]
    livewire.points = points
