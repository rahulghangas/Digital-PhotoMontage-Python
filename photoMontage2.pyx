import numpy as np
import cv2
import maxflow


class Data:
    images = list()
    offsets = list()
    draw = None
    gradient_x_sources = None
    gradient_y_sources = None
    height = None
    width = None
    source_constraints = None
    select_source = 0

    colors = list()

    def __init__(self):
        self.colors.append((255, 0, 0))
        self.colors.append((0, 200, 0))
        self.colors.append((0, 0, 255))
        self.colors.append((0, 190, 190))
        self.colors.append((180, 180, 0))
        self.colors.append((128, 0, 128))
        self.colors.append((100, 100, 100))


class Argument:
    data = Data()
    select_source = 0


def is_in_image(x, y, offset_x, offset_y, matrix_I):
    return 0 <= x - offset_x < matrix_I.shape[0] and 0 <= y - offset_y < matrix_I.shape[1]


class Collage:

    def __init__(self, data):
        self.D = data
        self._n_images = len(self.D.images)

    @property
    def n_images(self):
        return self._n_images

    @property
    def width(self):
        return self.D.width

    @property
    def height(self):
        return self.D.height

    def compute_photomontage(self, str outside_mode):
        cdef int width = self.D.width
        cdef int height = self.D.height
        cdef int size = len(self.D.sources)
        cdef int[:, :] result_label = np.zeros((height, width))
        cdef int i, j, k

        for i in range(height):
            for j in range(width):
                result_label[i, j] = 0


        cdef int min_cut = np.inf
        cdef int amelioration, skip_zero = True
        # /* Alpha expansion */

        cdef int alpha, flow
        while True:
            amelioration = False

            for alpha in len(size):
                if skip_zero and alpha == 0:
                    skip_zero = False
                    continue


                g = maxflow.Graph[float, float, float](3*height*width - width - height,
                                       2*height*width - width - height)

                # //cout << "Computing graph..." << endl;
                compute_graph(g, outside_mode,  result_label, self.D, alpha)
                # //cout << "Graph computed" << endl;
                flow = g.maxflow()
                if flow < min_cut:
                    amelioration = True
                    min_cut = flow

                for i in range(height):
                    for j in range(width):
                        if g.get_segment(width * i + j):
                                result_label[i, j] = alpha

            if not amelioration:
                break

        # /* Compute result image */
        cdef int[:,:] result_image = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                for k in range(size):
                    offset = self.D.offsets[k]
                    if result_label[i, j] == k:
                        result_image[i, j] = self.D.sources[k][i - offset[0], j - offset[1]]
                        if self.D.source_constraints[i, j] == 255:
                            self.D.draw[i, j] = (self.D.sources[k][i - offset[0], j - offset[1]] + self.D.colors[k % 7]) / 2
                        else:
                            self.D.draw[i, j] = self.D.colors[k % 7]

                        break

        cv2.imshow("Image", self.D.draw)
        cv2.imshow("Photomontage", result_image)
        cv2.waitKey()

def data_penalty(alpha, label, data, index_i, index_j):
    img_constraint = data.source_constraints[index_i, index_j]

    if not (label == alpha):
        if img_constraint == 255 or img_constraint == label or img_constraint == alpha:
            return 0

    return np.Inf


def interaction_penalty(pi, pj, qi, qj, data, label_p, label_q, outside_mode):
    offset_p = data.offsets[label_p]
    offset_q = data.offsets[label_q]
    matrix_slp = data.images[label_p]
    matrix_slq = data.images[label_q]
    matrix_gxlp = data.gradient_x_sources[label_p]
    matrix_gylp = data.gradient_y_sources[label_p]
    matrix_gxlq = data.gradient_x_sources[label_q]
    matrix_gylq = data.gradient_y_sources[label_q]

    if is_in_image(pi, pj, offset_q[0], offset_q[1], matrix_slq) or \
            is_in_image(qi, qj, offset_p[0], offset_p[1], matrix_slp) or \
            is_in_image(pi, pj, offset_p[0], offset_p[1], matrix_slp) or \
            is_in_image(qi, qj, offset_q[0], offset_q[1], matrix_slq):
        return 0

    if outside_mode == "COLORS":
        temp_p = np.linalg.norm(matrix_slp[pi - offset_p[0], pj - offset_p[1]])
    raise NotImplementedError()


def compute_graph(graph, str outside_mode, int[:,:] r_0, data, int alpha):
    cdef int height = data.height
    cdef int width = data.width

    graph.add_node(height * width)
    cdef int middle_node = height * width
    cdef double current_image, capacity_to_puits, capacity_to_source, \
        capacity_to_p, capacity_to_q
    cdef int currVoisin, i ,j


    for i in range(height):
        for j in range(width):
            current_image = r_0[i, j]
            capacity_to_puits = data_penalty(alpha, current_image, data, i, j)
            capacity_to_source = np.inf
            if i - data.offsets[alpha][0] >= 0 and j - data.offsets[alpha][1] >= 0:
                capacity_to_source = data_penalty(alpha, alpha, data, i, j)
            

            graph.add_tweights(width * i + j, capacity_to_source, capacity_to_puits)

            if i < height - 1:
                currVoisin = r_0[i + 1, j]
                graph.add_node(1)
                capacity_to_puits = interaction_penalty(i, j, i + 1, j, data, current_image, currVoisin, outside_mode)
                graph.add_tweights(middle_node, 0, capacity_to_puits)
                capacity_to_p = interaction_penalty(i, j, i + 1, j, data, current_image, alpha, outside_mode)
                capacity_to_q = interaction_penalty(i, j, i + 1, j, data, alpha, currVoisin, outside_mode)
                graph.add_edge(width * i + j, middle_node, capacity_to_p, capacity_to_p)
                graph.add_edge(width * (i + 1) + j, middle_node, capacity_to_q, capacity_to_q)
                middle_node += 1
            

            if j < width - 1:
                currVoisin = r_0[i, j + 1]
                graph.add_node(1)
                capacity_to_puits = interaction_penalty(i, j, i, j + 1, data, current_image, currVoisin, outside_mode)
                if current_image != currVoisin: capacity_to_puits += 1
                graph.add_tweights(middle_node, 0, capacity_to_puits)
                capacity_to_p = interaction_penalty(i, j, i, j + 1, data, current_image, alpha, outside_mode)
                capacity_to_q = interaction_penalty(i, j, i, j + 1, data, alpha, currVoisin, outside_mode)
                graph.add_edge(width * i + j, middle_node, capacity_to_p, capacity_to_p)
                graph.add_edge(width * i + j + 1, middle_node, capacity_to_q, capacity_to_q)
                middle_node += 1

win_name = "Image"

def compute_photomontage(data):
    C = Collage(data)
    C.compute_photomontage("COLORS_AND_GRADIENTS")

def on_mouse(event, x, y, foo, data):
    if (foo == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_MOUSEMOVE):
        cv2.circle(data.draw, (x, y), 2, data.colors[data.select_source], 2)
        cv2.circle(data.source_constraints, (x, y), 2, data.select_source, 2)
        cv2.imshow(win_name, data.draw)
    elif (event == cv2.EVENT_RBUTTONDOWN):
        data.select_source = data.select_source + 1
        