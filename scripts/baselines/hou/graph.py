import numpy as np

'''
For Paper:
https://doi.org/10.1109/AVSS52988.2021.9663770.
'''
num_nodes_face_hou = 43 # 23
num_nodes_body_hou = 11

# from openpose 25 to Hou et al. 2021 11 nodes
body_inward_edges_hou = [(9,2), (10,2), (2,1), (0,1), (3,1), (6,1),
                         (8,7), (7,6), (5,4), (4,3)    
                        ]
# from openpose 25 to Hou et al. 2021 23 nodes
face_inward_edges_hou = [
               (27,28),(28,29),(29,30),   #nose upper part
               (31,32),(32,33),(33,34),(34,35), #nose lower part
               (36,37),(37,38),(38,39),(39,40),(40,41),(41,36), #right eye
               (42,43),(43,44),(44,45),(45,46),(46,47),(47,42), #left eye
               (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),(54,55),(55,56),
               (56,57),(57,58),(58,59),(59,48), #Lip outline
               (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67),(67,60) #Lip inner line 
    ]
# subtract 27 from face_inward_edges_hou as we removed the outline form openpose face keypoints
# to match the HOu et al. 2021 23 nodes
face_inward_edges_hou = [(i-27, j-27) for (i, j) in face_inward_edges_hou]

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
