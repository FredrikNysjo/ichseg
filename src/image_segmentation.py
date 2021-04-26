import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


MPR_PLANE_X = 0
MPR_PLANE_Y = 1
MPR_PLANE_Z = 2
TOOL_MODE_2D = 0
TOOL_MODE_3D = 1


class BrushTool:
    def __init__(self):
        self.position = glm.vec4(0.0)
        self.size = 30
        self.enabled = False
        self.painting = False
        self.count = 0
        self.mode = TOOL_MODE_3D
        self.plane = MPR_PLANE_Z


class PolygonTool:
    def __init__(self):
        self.points = []
        self.enabled = False
        self.rasterise = False
        self.clicking = False
        self.plane = MPR_PLANE_Z


class LivewireTool:
    def __init__(self):
        self.graph = None
        self.pred = None
        self.dist = None
        self.path = []
        self.points = []
        self.smoothing = True
        self.enabled = False
        self.rasterise = False
        self.clicking = False
        self.plane = MPR_PLANE_Z


class SmartBrushTool:
    def __init__(self):
        self.position = glm.vec4(0.0)
        self.size = 30
        self.sensitivity = 5.0
        self.delta_scaling = 1.0
        self.enabled = False
        self.painting = False
        self.momentum = 0
        self.xy = (0, 0)
        self.count = 0
        self.mode = TOOL_MODE_3D
        self.plane = MPR_PLANE_Z


class SeedPaintTool:
    def __init__(self):
        self.enabled = False
        self.plane = MPR_PLANE_Z


class ToolManager:
    def __init__(self):
        self.brush = BrushTool()
        self.polygon = PolygonTool()
        self.livewire = LivewireTool()
        self.smartbrush = SmartBrushTool()
        self.seedpaint = SeedPaintTool()


def tools_disable_all_except(tools, selected) -> None:
    """ Disable all tools except the selected one (provided as reference) """
    tools.polygon.enabled = False
    tools.brush.enabled = False
    tools.livewire.enabled = False
    tools.smartbrush.enabled = False
    tools.seedpaint.enabled = False
    selected.enabled = True
    tools_cancel_drawing_all(tools)


def tools_cancel_drawing_all(tools) -> None:
    """ Cancel drawing for all tools """
    tools.polygon.points = []
    tools.livewire.path = []
    tools.livewire.points = []


def tools_set_plane_all(tools, axis) -> None:
    """ Set the active drawing plane for all tools

    This also cancels all drawing, to prevent the user from
    continue a polygon or livewire on another plane.
    """
    tools.polygon.plane = axis
    tools.brush.plane = axis
    tools.livewire.plane = axis
    tools.smartbrush.plane = axis
    tools.seedpaint.plane = axis
    tools_cancel_drawing_all(tools)


def brush_tool_apply(tool, image, texcoord, spacing):
    """ Apply brush tool to input 3D image

    Returns result (subimage, offset) if successfull, otherwise None
    """
    if abs(texcoord.x - 0.5) > 0.5 or abs(texcoord.y - 0.5) > 0.5 or abs(texcoord.z - 0.5) > 0.5:
        return None
    return _brush_tool_apply(tool, image, texcoord, spacing)


def _brush_tool_apply(tool, image, texcoord, spacing):
    """ Apply brush tool to input 3D image (helper function) """
    d, h, w = image.shape[0:3]
    center = texcoord * glm.vec3(w, h, d)
    radius = glm.vec3(tool.size)
    if tool.mode == TOOL_MODE_2D:
        radius[tool.plane] = 1  # Restrict brush to 2D plane
    lower = center - glm.vec3(radius * (spacing.x / spacing)) * 0.5 + 0.5
    upper = center + glm.vec3(radius * (spacing.x / spacing)) * 0.5 + 0.5
    lower = glm.ivec3(glm.clamp(lower, glm.vec3(0.0), glm.vec3(w - 1, h - 1, d - 1)))
    upper = glm.ivec3(glm.clamp(upper, glm.vec3(0.0), glm.vec3(w - 1, h - 1, d - 1)))

    xx = np.arange(lower.x - center.x + 0.5, upper.x - center.x + 0.5, 1.0)
    yy = np.arange(lower.y - center.y + 0.5, upper.y - center.y + 0.5, 1.0)
    zz = np.arange(lower.z - center.z + 0.5, upper.z - center.z + 0.5, 1.0)
    z, x, y = np.meshgrid(zz, yy, xx, indexing='ij')
    z *= spacing.z / spacing.x;  # Need to take slice spacing into account
    subimage = (tool.size*0.5 - (x**2 + y**2 + z**2)**0.5 + 0.5) * 255.99
    subimage = np.maximum(0, np.minimum(255, subimage)).astype(dtype=np.uint8)

    image[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x] = np.maximum(
        image[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x], subimage)
    subimage = image[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x]
    return subimage, lower


def smartbrush_tool_apply(tool, image, volume, texcoord, spacing, level_range=None):
    """ Apply smart brush to input 3D image

    This tool uses the SmartPaint method from [Malmberg et al. 2014].
    Reference: F. Malmberg et al., "SmartPaint: a tool for interactive
    segmentation of medical volume images", CMBBE, 2014.

    Returns result (subimage, offset) if successfull, otherwise None
    """
    if abs(texcoord.x - 0.5) > 0.5 or abs(texcoord.y - 0.5) > 0.5 or abs(texcoord.z - 0.5) > 0.5:
        return None
    return _smartbrush_tool_apply(tool, image, volume, texcoord, spacing, level_range)


def _smartbrush_tool_apply(tool, image, volume, texcoord, spacing, level_range):
    """ Apply smart brush to input 3D image (helper function)

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
    lower = glm.ivec3(glm.clamp(lower, glm.vec3(1.0), glm.vec3(w - 2, h - 2, d - 2)))
    upper = glm.ivec3(glm.clamp(upper, glm.vec3(1.0), glm.vec3(w - 2, h - 2, d - 2)))

    xx = np.arange(lower.x - center.x + 0.5, upper.x - center.x + 0.5, 1.0)
    yy = np.arange(lower.y - center.y + 0.5, upper.y - center.y + 0.5, 1.0)
    zz = np.arange(lower.z - center.z + 0.5, upper.z - center.z + 0.5, 1.0)
    z, x, y = np.meshgrid(zz, yy, xx, indexing='ij')
    z *= spacing.z / spacing.x;  # Need to take slice spacing into account

    shift = level_range[0]
    scale = 1.0 / (level_range[1] - level_range[0])
    subimage = image[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x].astype(dtype=np.float32)
    midpoint = volume[int(center.z),int(center.y),int(center.x)]
    midpoint = (midpoint - shift) * scale
    intensity = volume[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x].astype(dtype=np.float32)
    intensity -= shift
    intensity *= scale

    value = subimage * (1.0 / 255.0)
    sigma = np.maximum(0.0, 1.0 - (x**2 + y**2 + z**2)**0.5 / (tool.size * 0.5))
    delta = np.abs(intensity - midpoint) * tool.delta_scaling
    rho = np.maximum(0.0, 1.0 - delta)**tool.sensitivity
    alpha = sigma * rho
    subimage[:,:,:] = (0.05 * alpha + (1.0 - 0.05 * alpha) * value) * 255.0

    subimage = np.maximum(0.0, np.minimum(255.0, subimage)).astype(dtype=np.uint8)
    image[lower.z:upper.z,lower.y:upper.y,lower.x:upper.x] = subimage
    return subimage, lower


def rasterise_polygon_2d(polygon, image) -> np.array:
    """ Rasterise closed 2D polygon using fast XOR-based method """
    h, w = image.shape
    xmin = int(np.floor(np.min(polygon[:,0]) * w))
    xmax = int(np.ceil(np.max(polygon[:,0]) * w))
    ymin = int(np.floor(np.min(polygon[:,1]) * h))
    ymax = int(np.ceil(np.max(polygon[:,1]) * h))
    output = np.zeros((h, w), dtype=np.uint8)
    nvertices = polygon.shape[0]
    for y in range(ymin, ymax):
        for i in range(0, nvertices - 1):
            v0 = polygon[i + 0,:] * (w, h)
            v1 = polygon[i + 1,:] * (w, h)
            if v1[0] < v0[0]:
                v0, v1 = v1, v0  # Swap vertices s.t. v0.x <= v1.x
            delta = v1 - v0
            t = ((y + 0.5) - v0[1]) / delta[1] if delta[1] else 0.0
            if t > 0.0 and t < 1.0:
                x = v0[0] + t * delta[0]
                output[int(y + 0.5), 0:min(int(x + 0.5),w)] ^= 255
                output[int(y + 0.5), min(int(x + 0.5),w):w] ^= 0
    return output


def polygon_tool_apply(tool, image):
    """ Apply polygon tool to 2D slice of input 3D image

    Returns result (subimage, offset) if successfull, otherwise None
    """
    if len(tool.points) == 0:
        return None

    # Extract image slice for current drawing plane
    d, h, w = image.shape
    if tool.plane == MPR_PLANE_Z:
        offset = (0, 0, int((tool.points[2] + 0.5) * d))
        slice_ = image[offset[2],:,:].astype(dtype=np.uint8)
        axes = (0, 1, 2)
        output_shape = (1, h, w)
    elif tool.plane == MPR_PLANE_Y:
        offset = (0, int((tool.points[1] + 0.5) * h), 0)
        slice_ = image[:,offset[1],:].astype(dtype=np.uint8)
        axes = (0, 2, 1)
        output_shape = (d, 1, w)
    elif tool.plane == MPR_PLANE_X:
        offset = (int((tool.points[0] + 0.5) * w), 0, 0)
        slice_ = image[:,:,offset[0]].astype(dtype=np.uint8)
        axes = (1, 2, 0)
        output_shape = (d, h, 1)
    else:
        assert False, "Invalid MPR plane index"

    # Construct closed 2D polygon from drawn 3D points
    npoints = len(tool.points) // 3
    polygon = np.zeros((npoints + 1, 2))
    for i in range(0, npoints):
        polygon[i, 0] = tool.points[3 * i + axes[0]] + 0.5
        polygon[i, 1] = tool.points[3 * i + axes[1]] + 0.5
    polygon[npoints, 0] = tool.points[axes[0]] + 0.5
    polygon[npoints, 1] = tool.points[axes[1]] + 0.5

    # Rasterise 2D polygon into image slice
    subimage = rasterise_polygon_2d(polygon, slice_)
    subimage |= slice_                         # Merge with previous result
    subimage = subimage.reshape(output_shape)  # Final result must be a volume

    return subimage, offset


def livewire_tool_apply(tool, image):
    """ Apply livewire tool to 2D slice of input 3D image

    Returns result (subimage, offset) if successfull, otherwise None
    """
    # Re-use the existing code for the polygon tool, since a livewire is
    # basically just a polygon with a vertex for each pixel or voxel
    return polygon_tool_apply(tool, image)


def create_graph_from_image(image):
    """ Constructs a sparse matrix for a 4-connected image graph """
    h, w = image.shape
    weights = np.array([1., 1., 1., 1.])
    offsets = (-w, -1, 1, w)
    graph = sp.sparse.diags(weights, offsets, shape=(w*h, w*h), format='csr')
    return graph


def update_edge_weights(graph, image, alpha0, alpha1):
    """ Update graph edge weights from image values """
    assert(graph.format == 'csr' or graph.format == 'lil')
    h, w = image.shape
    for y in range(1, h - 1):
        diff0 = image[y-1, 1:w-1] - image[y, 1:w-1]
        diff1 = image[y+0, 0:w-2] - image[y, 1:w-1]
        diff2 = image[y+0, 2:w-0] - image[y, 1:w-1]
        diff3 = image[y+1, 1:w-1] - image[y, 1:w-1]
        grad0 = image[y-1,0:w-2] + image[y,0:w-2] - image[y-1,2:w] - image[y,2:w]
        grad1 = image[y-1,0:w-2] + image[y-1,1:w-1] - image[y+1,0:w-2] - image[y+1,1:w-1]
        grad2 = image[y-1,1:w-1] + image[y-1,2:w] - image[y+1,1:w-1] - image[y+1,2:w]
        grad3 = image[y+1,0:w-2] + image[y,0:w-2] - image[y+1,2:w] - image[y,2:w]
        w0 = 1.0 / (alpha0 * np.abs(diff0) + alpha1 * np.abs(grad0) + 1e-3)
        w1 = 1.0 / (alpha0 * np.abs(diff1) + alpha1 * np.abs(grad1) + 1e-3)
        w2 = 1.0 / (alpha0 * np.abs(diff2) + alpha1 * np.abs(grad2) + 1e-3)
        w3 = 1.0 / (alpha0 * np.abs(diff3) + alpha1 * np.abs(grad3) + 1e-3)
        for x in range(1, w - 1):
            idx = y * w + x
            # Updating the matrix row data directly is much faster than
            # accessing indivdual elements
            if graph.format == 'lil':
                graph.data[idx] = (w0[x-1], w1[x-1], w2[x-1], w3[x-1])
            if graph.format == 'csr':
                graph.data[graph.indptr[idx]:graph.indptr[idx+1]] = (w0[x-1], w1[x-1], w2[x-1], w3[x-1])


def compute_dijkstra(graph, seed):
    """ Computes distances to seed point(s) in the graph, and
        predecessor matrix, using Dijkstra's algorithm
    """
    dist, pred = sp.sparse.csgraph.dijkstra(graph, directed=False, indices=seed, return_predecessors=True)
    return dist, pred


def compute_shortest_path(pred, a, b):
    """ Compute shortest path (as index list) from predecessor matrix """
    path = []
    idx = pred[b]
    while idx != -9999 and idx != a:
        path.append(idx)
        idx = pred[idx]
    path.reverse()
    return path


def update_livewire(livewire, path_new, z, volume):
    """ Update line segments of livewire from its current path and
        new path not yet appended to the livewire (for preview)
    """
    d, h, w = volume.shape
    points = []
    for idx in (livewire.path + path_new):
        x = (idx % w) / float(w) - 0.5
        y = (idx // w) / float(w) - 0.5
        points.extend((x, y, z))
    livewire.points = points


def smooth_livewire(livewire, iterations=5):
    """ Apply smoothing to line segments in livewire """
    points = livewire.points
    output = [x for x in points]
    for j in range(0, iterations):
        for i in range(3, len(points) - 3):
            output[i] = (points[i - 3] + points[i + 3]) * 0.25 + points[i] * 0.5
        points = [x for x in output]
    livewire.points = points
