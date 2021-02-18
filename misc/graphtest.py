import numpy as np
import scipy as sp
import scipy.misc
import scipy.sparse
import scipy.sparse.csgraph
import matplotlib.pyplot as plt
import time


def create_graph_from_image(image):
    h, w = image.shape
    weights = np.array([1., 1., 1., 1.])
    offsets = (-w, -1, 1, w)
    graph = sp.sparse.diags(weights, offsets, shape=(w*h, w*h), format='csr')
    return graph


def update_edge_weights(image, graph):
    """ Update graph edge weights from image values """
    assert(graph.format == 'csr' or graph.format == 'lil')
    h, w = image.shape
    for y in range(1, h - 1):
        w0 = np.abs(image[y-1, 1:w-1] - image[y, 1:w-1]) + 1e-3
        w1 = np.abs(image[y+0, 0:w-2] - image[y, 1:w-1]) + 1e-3
        w2 = np.abs(image[y+0, 2:w-0] - image[y, 1:w-1]) + 1e-3
        w3 = np.abs(image[y+1, 1:w-1] - image[y, 1:w-1]) + 1e-3
        for x in range(1, w - 1):
            idx = y * w + x
            # Updating the matrix row data directly is much faster than
            # accessing indivdual elements
            if graph.format == 'lil':
                graph.data[idx] = (w0[x-1], w1[x-1], w2[x-1], w3[x-1])
            if graph.format == 'csr':
                graph.data[graph.indptr[idx]:graph.indptr[idx+1]] = (w0[x-1], w1[x-1], w2[x-1], w3[x-1])


def compute_dijkstra(graph, seed):
    dist, pred = sp.sparse.csgraph.dijkstra(graph, directed=False, indices=seed, return_predecessors=True)
    return dist, pred


def compute_shortest_path(pred, a, b):
    """ Compute shortest path (as list of indices) from predecessor matrix """
    path = [b]
    idx = pred[b]
    while idx != -9999 and idx != a:
        path.append(idx)
        idx = pred[idx]
    return path


image = sp.misc.ascent()
#image = image[::2,::2]  # Downsample image
h, w = image.shape

print("Constructing graph...")
tic = time.time()
graph = create_graph_from_image(image)
print("Done (%f s)" % (time.time() - tic))

print("Updating graph weights from image...")
tic = time.time()
update_edge_weights(image, graph)
print("Done (%f s)" % (time.time() - tic))

print("Computing Dijkstra's algorithm...")
tic = time.time()
p = (int(w * 0.75), int(h * 0.5))
q = (int(w * 0.25), int(h * 0.5))
p_idx = p[1] * w + p[0]
q_idx = q[1] * w + q[0]
dist, pred = compute_dijkstra(graph, p_idx)
print("Done (%f s)" % (time.time() - tic))

print("Computing shortest path...")
tic = time.time()
path = compute_shortest_path(pred, p_idx, q_idx)
xx = [index % w for index in path]
yy = [index // w for index in path]
print("Done (%f s)" % (time.time() - tic))

plt.figure()
plt.imshow(image, cmap='gray')
plt.figure()
plt.imshow(dist.reshape((h, w)))
plt.plot(xx, yy, '-r')
plt.plot(p[0], p[1], 'or')
plt.plot(q[0], q[1], 'or')
plt.colorbar()
plt.show()
