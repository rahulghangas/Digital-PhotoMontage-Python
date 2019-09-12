import numpy as np
import maxflow
import sys
import math
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline int color_diff(int[:,:,:] img_a, int[:, :,:] img_b, int x, int y):
    cdef int d = 0
    cdef int c

    for c in range(3):
        d += (img_a[y, x, c] - img_b[y, x, c]) ** 2
    return math.sqrt(d)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def solve(int [:,:,:,:] photos, mask):
    cdef double e = sys.float_info.epsilon

    assignment = np.zeros(mask.shape, dtype=np.intc)


    assignment[mask != -1] = mask[ mask != -1]

    cdef Py_ssize_t rows = mask.shape[0]
    cdef Py_ssize_t cols = mask.shape[1]
    cdef int [:,:]mask_view = mask
    cdef int [:,:]assignment_view

    graph = maxflow.Graph[float]()

    converged = False

    cdef int x,y, u, v, label_u, label_v, var_idx
    cdef double var_a, var_b, var_c, var_d, delta, energy
    cdef double inf = np.inf
    cdef double min_energy = inf

    while not converged:
        assignment_view = assignment
        converged = True

        for alpha in range(4):
            print("At alpha %d", alpha)

            graph.reset()

            nodeids = graph.add_grid_nodes(rows * cols)

            # sink_edges = np.zeros((rows, cols))
            # sink_edges[(mask != -1) & (mask != alpha)] = np.inf
            # graph.add_grid_tedges(nodeids, 0, sink_edges.flatten())
            var_idx = 0
            for y in range(rows):
                for x in range(cols):
                    if mask_view[y, x] != -1 and mask_view[y, x] != alpha:
                        graph.add_tedge(var_idx, 0, inf)
                    
                    var_idx += 1

            # u = np.arange(1, rows * (cols -1) + 1).reshape((rows, cols - 1))
            # v = u - 1
            #
            # label_u = assignment[:, 1:]
            # label_v = assignment[:, :-1]


            # Convert loops to vectorised operations
            for y in range(rows):
                for x in range(1, cols):
                    u = y * cols + x
                    v = u - 1

                    label_u = assignment_view[y,x]
                    label_v = assignment_view[y, x-1]

                    if label_u == alpha and label_v == alpha:
                        continue

                    var_a = 0.0
                    var_b = color_diff(photos[alpha], photos[label_v], x,y) + \
                        color_diff(photos[alpha], photos[label_v], x-1,y)
                    var_c = color_diff(photos[label_u], photos[alpha], x,y) + \
                        color_diff(photos[label_u], photos[alpha], x-1,y)
                    var_d = color_diff(photos[label_u], photos[label_v], x,y) + \
                        color_diff(photos[label_u], photos[label_v], x-1,y)

                    if var_a + var_d > var_c + var_b:
                        delta = var_a + var_d - var_c -var_b
                        var_a -= delta/3 - e
                        var_c += delta/3 + e
                        var_b = var_a + var_d -var_c + e

                    graph.add_tedge(u, var_d, var_a)

                    var_b -= var_a
                    var_c -= var_d
                    var_b += e
                    var_c += e

                    if var_b < 0:
                        graph.add_tedge(u, 0, var_b)
                        graph.add_tedge(v, 0, -var_b)
                        graph.add_edge(u, v, 0.0, var_b + var_c)

                    elif var_c < 0:
                        graph.add_tedge(u, 0, -var_c)
                        graph.add_tedge(v, 0, var_c)
                        graph.add_edge(u, v, var_b + var_c, 0.0)

                    else:
                        graph.add_edge(u, v, var_b, var_c)


            for y in range(1, rows):
                for x in range(cols):
                    u = y * cols + x
                    v = (y-1) * cols + x

                    label_u = assignment_view[y,x]
                    label_v = assignment_view[y - 1, x]

                    if label_u == alpha and label_v == alpha:
                        continue

                    var_a = 0.0
                    var_b = color_diff(photos[alpha], photos[label_v],x,y) + \
                        color_diff(photos[alpha], photos[label_v], x, y-1)
                    var_c = color_diff(photos[label_u], photos[alpha], x,y) + \
                        color_diff(photos[label_u], photos[alpha], x, y-1)
                    var_d = color_diff(photos[label_u], photos[label_v], x,y) + \
                        color_diff(photos[label_u], photos[label_v], x, y-1)

                    if var_a + var_d > var_c + var_b:
                        delta = var_a + var_d - var_c -var_b
                        var_a -= delta/3 - e
                        var_c += delta/3 + e
                        var_b = var_a + var_d -var_c + e

                    graph.add_tedge(u, var_d, var_a)

                    var_b -= var_a
                    var_c -= var_d
                    var_b += e
                    var_c += e

                    if var_b < 0:
                        graph.add_tedge(u, 0, var_b)
                        graph.add_tedge(v, 0, -var_b)
                        graph.add_edge(u, v, 0.0, var_b + var_c)

                    elif var_c < 0:
                        graph.add_tedge(u, 0, -var_c)
                        graph.add_tedge(v, 0, var_c)
                        graph.add_edge(u, v, var_b + var_c, 0.0)

                    else:
                        graph.add_edge(u, v, var_b, var_c)

            energy = graph.maxflow()

            if energy < min_energy:
                min_energy = energy
                converged = False

                sgm = graph.get_grid_segments(nodeids)
                assignment[~sgm.reshape(assignment.shape)] = alpha

    return assignment
