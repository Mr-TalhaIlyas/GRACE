import numpy as np
# Nods and grpahs follow the same convention as in mmaction2
# https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/utils/graph.py
num_nodes_face = 70
num_nodes_hand = 21
num_nodes_body = 25

body_inward_edges = [(4,3), (3,2), (7,6), (6,5), (5,1), (2,1),
                    (0,1),(16,0),(15,0), (18,16), (17,15),
                    (8,1), (12,8), (9,8), (13,12), (10,9), (14,13), (11,10),
                    (24,11), (21,14), (22,11), (19,14), (20,19), (23,22)]

hand_inward_edges = [(4,3), (3,2), (2,1), (1,0),
                     (8,7), (7,6), (6,5), (5,0),
                     (12,11), (11,10), (10,9), (9,0),
                     (16,15), (15,14), (14,13), (13,0),
                     (20,19), (19,18), (18,17), (17,0)]

face_inward_edges = [
               (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),
               (8,9),(9,10),(11,12),(12,13),(14,15),(15,16), #outline 
               (17,18),(18,19),(19,20),(20,21), #right eyebrow
               (22,23),(23,24),(24,25),(25,26), #left eyebrow
               (27,28),(28,29),(29,30),   #nose upper part
               (31,32),(32,33),(33,34),(34,35), #nose lower part
               (36,37),(37,38),(38,39),(39,40),(40,41),(41,36), #right eye
               (42,43),(43,44),(44,45),(45,46),(46,47),(47,42), #left eye
               (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),(54,55),(55,56),
               (56,57),(57,58),(58,59),(59,48), #Lip outline
               (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67),(67,60) #Lip inner line 
               ]

# https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/utils/graph.py#L167
coco_num_node = 17
coco_inward_edges = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
               (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
               (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]

ntu_rgbd_nodes = 25
nturgbd_inwards = [ (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                    (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                    (17, 1), (18, 17), (19, 18), (20, 19), (22, 8),
                    (23, 8), (24, 12), (25, 12)]
nturgbd_inward = [(i - 1, j - 1) for (i, j) in nturgbd_inwards] # so that starting node is zero like others

openpose_18 = 18
openpose_inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                   (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                   (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A


class Graph:
    def __init__(self, num_nodes, inward_edges, labeling_mode='spatial'):
        self.num_node = num_nodes
        self.inward = inward_edges
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = get_uniform_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'distance*':
            A = get_uniform_distance_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'distance':
            A = get_distance_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'spatial':
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        elif labeling_mode == 'DAD':
            A = get_DAD_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'DLD':
            A = get_DLD_graph(self.num_node, self.self_link, self.neighbor)
        else:
            raise ValueError("Unknown labeling mode: {}".format(labeling_mode))
        return A
