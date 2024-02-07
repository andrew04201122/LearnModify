import networkx as nx
import pickle
import random
import math
from os import path
from heapdict import heapdict
import numpy as np  
def pickle_save(data,file_name):
    if path.exists(file_name):
        raise BaseException('file already exists')

    with open(file_name,'wb') as f:
        pickle.dump(data , f)

def pickle_load(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)

def create_one_graph(n,p):
    g = nx.erdos_renyi_graph(n , p)
    return g

def greedy2approx(graph):
    """this function is an approximate method"""
    covered_edge = 0
    num_edge = len(graph.edges())
    hd = heapdict()
    degree = nx.degree(graph)
    for v , d in degree:
        hd[v] = -d
        
    select_nodes = set()
    
    while covered_edge < num_edge:
        cur_v , cur_deg = hd.popitem()
        select_nodes.add(cur_v)
        covered_edge += -(cur_deg)
        for u in graph.neighbors(cur_v):
            if u not in select_nodes:
                hd[u] += 1
    return select_nodes

def mvc_bb(graph , UB = 9999999 , C = []): #heuristic method
    if len(graph.edges()) == 0:
        return C
    LB = 0 #lower bound
    # if lower bound add current cover number larger than upper bound, it means there is impossible to find the better result
    if len(C) + LB >= UB: 
        return [i for i in range(UB+1)]
    
    degree_list = nx.degree(graph)
    v , d = max(degree_list , key = lambda a : a[1])
    #use branch and bound to find out the mvc result
    #C1 will include all neighbor node of v
    #C2 will only include v
    C1 = C[:] 
    C2 = C[:]
    graph_1 = graph.copy()
    C1.extend(list(graph.neighbors(v)))
    graph_1.remove_nodes_from(C1)
    C1= mvc_bb(graph_1 , UB , C1)

    C2.append(v)
    graph_2 = graph.copy()
    graph_2.remove_node(v)
    C2 = mvc_bb(graph_2 , min(UB , len(C1)) , C2 )

    if len(C1)>len(C2):
        return C2
    else:
        return C1
    
def get_gcn_input(g):
    """this function is used to get the normalized graph laplacian matrix, which is common used in GCN"""
    A = nx.convert_matrix.to_numpy_array(g)
    D = np.diag(np.sum(A , 0))
    D_2 = D ** -0.5
    D_2[np.isinf(D_2)] = 0
    normalized_graph_lap = np.matmul(np.matmul(D_2 , A) , D_2) + np.eye(A.shape[0])
    return normalized_graph_lap

