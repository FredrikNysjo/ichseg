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
        self.count = 0


class PolygonTool:
    def __init__(self):
        self.points = []
        self.enabled = False
        self.rasterise = False
        self.clicking = False


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


class SmartBrushTool:
    def __init__(self):
        self.position = glm.vec4(0.0)
        self.size = 30
        self.sensitivity = 5.0
        self.delta_scaling = 50.0
        self.enabled = False
        self.painting = False
        self.momentum = 0
        self.xy = (0, 0)
        self.count = 0


def apply_brush(image, texcoord, brush):
    """ Apply brush to image """
    if abs(texcoord.x - 0.5) > 0.5 or abs(texcoord.y - 0.5) > 0.5 or abs(texcoord.z - 0.5) > 0.5:
        return
    d, h, w = image.shape[0:3]
    center = texcoord * glm.vec3(w, h, d)
    lower = center - glm.vec3(brush.size, brush.size, 0.0) * 0.5
    upper = center + glm.vec3(brush.size, brush.size, 0.0) * 0.5
    lower = glm.ivec3(glm.clamp(lower, glm.vec3(0.0), glm.vec3(w - 1, h - 1, d - 1)))
    upper = glm.ivec3(glm.clamp(upper, glm.vec3(0.0), glm.vec3(w - 1, h - 1, d - 1)))
    subimage = np.ones((1, upper.y - lower.y, upper.x - lower.x), dtype=np.uint8) * 255
    xx = np.arange(glm.floor(lower.x - center.x + 0.5), glm.floor(upper.x - center.x + 0.5), 1.0)
    yy = np.arange(glm.floor(lower.y - center.y + 0.5), glm.floor(upper.y - center.y + 0.5), 1.0)
    x, y = np.meshgrid(xx, yy)
    subimage[0,:,:] = ((x**2 + y**2)**0.5 < brush.size*0.5).astype(dtype=np.uint8) * 255
    image[lower.z:upper.z+1,lower.y:upper.y,lower.x:upper.x] |= subimage
    subimage = image[lower.z:upper.z+1,lower.y:upper.y,lower.x:upper.x]
    return subimage, lower


def apply_smartbrush(image, volume, texcoord, brush):
    """ Apply smart brush to image, using the method from SmartPaint.

        Reference: F. Malmberg et al., "SmartPaint: a tool for
        interactive segmentation of medical volume images", CMBBE, 2014.
    """
    if abs(texcoord.x - 0.5) > 0.5 or abs(texcoord.y - 0.5) > 0.5 or abs(texcoord.z - 0.5) > 0.5:
        return
    d, h, w = image.shape[0:3]
    center = texcoord * glm.vec3(w, h, d)
    lower = center - glm.vec3(brush.size, brush.size, 0.0) * 0.5
    upper = center + glm.vec3(brush.size, brush.size, 0.0) * 0.5
    lower = glm.ivec3(glm.clamp(lower, glm.vec3(1.0), glm.vec3(w - 2, h - 2, d - 2)))
    upper = glm.ivec3(glm.clamp(upper, glm.vec3(1.0), glm.vec3(w - 2, h - 2, d - 2)))
    subimage = image[lower.z:upper.z+1,:,:].astype(dtype=np.float32)
    midpoint = float(volume[lower.z,int(center.y),int(center.x)]) / 32768.0
    for y in range(lower.y, upper.y):
        for x in range(lower.x, upper.x):
            intensity = float(volume[lower.z,y,x]) / 32768.0
            value = float(image[lower.z,y,x]) / 255.0
            sigma = max(1.0 - ((x - center.x)**2 + (y - center.y)**2)**0.5 / (brush.size * 0.5), 0.0)
            rho = max(1.0 - brush.delta_scaling * abs(intensity - midpoint), 0.0)**brush.sensitivity
            alpha = sigma * rho
            subimage[0,y,x] = (0.05 * alpha + (1.0 - 0.05 * alpha) * value) * 255.0
    subimage = np.maximum(0.0, np.minimum(255.0, subimage)).astype(dtype=np.uint8)
    image[lower.z:upper.z+1,:,:] = subimage
    return subimage, (0, 0, lower.z)


def rasterise_polygon(polygon, image, zoffset) -> np.array:
    """ Rasterise closed 2D polygon using fast XOR-based method """
    d, h, w = image.shape
    xmin = int(np.floor(np.min(polygon[:,0]) * w))
    xmax = int(np.ceil(np.max(polygon[:,0]) * w))
    ymin = int(np.floor(np.min(polygon[:,1]) * h))
    ymax = int(np.ceil(np.max(polygon[:,1]) * h))
    output = np.zeros((1, h, w), dtype=np.uint8)
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
                output[0,int(y + 0.5),0:min(int(x + 0.5),w)] ^= 255
                output[0,int(y + 0.5),min(int(x + 0.5),w):w] ^= 0
    return output


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
