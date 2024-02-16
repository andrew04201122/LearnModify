import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import random
import argparse
from numpy.linalg import norm

from utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--iteration", default= 100, type=int)
parser.add_argument("--modify_edge", default= 400, type=int)
args = parser.parse_args()

edge_dict = {}
whole_edge_set = set()
def calculate_MVC(graph, UB=9999999, C=set()):
    """use branch and bound to find out the mvc result"""
    if len(graph.edges()) == 0:
        return C

    v, _ = max(graph.degree(), key=lambda a: a[1])

    # C1 分支：選擇鄰居
    C1 = C.copy()
    neighbors = set(graph.neighbors(v))
    C1.update(neighbors)
    graph_1 = graph.copy()
    graph_1.remove_nodes_from(neighbors)
    if len(C1) < UB:
        C1 = calculate_MVC(graph_1, UB, C1)

    # C2 分支：只選擇該節點
    C2 = C.copy()
    C2.add(v)
    graph_2 = graph.copy()
    graph_2.remove_node(v)
    if len(C2) < UB:
        C2 = calculate_MVC(graph_2, min(UB, len(C1)), C2)

    return min(C1, C2, key=len)

def create_edge_dict(graph_size):
    index = 0
    for i in range(graph_size - 1):
        for j in range(i + 1, graph_size):
            whole_edge_set.add((i, j))
            edge_dict[(i, j)] = index
            index += 1

if __name__ == "__main__":
    testing_graphs, opt, pro = pickle_load("/workspace/Synthetic_graph/Validation_graph_200_withOPTPRO.pkl")
    iteration = args.iteration
    modify_edge = args.modify_edge  
    graph_size = 50
    create_edge_dict(graph_size)
    
    mvc_ans = []
    for i in range(len(testing_graphs)):
        print(i)
        A1 = nx.adjacency_matrix(testing_graphs[i]).toarray()
        candidate_modify_edge = random.sample(whole_edge_set, modify_edge)
        modify_count = 0
        for v in candidate_modify_edge:
            # modify_pro = random.random()
            # bernoulli = torch.distributions.Bernoulli(modify_pro)
            # decisions = bernoulli.sample().item()
            decisions = 1
            if decisions == 1:
                modify_count += 1
                if testing_graphs[i].has_edge(v[0], v[1]):
                    testing_graphs[i].remove_edge(v[0], v[1])
                else:
                    testing_graphs[i].add_edge(v[0], v[1])
        mvc_ans.append(len(calculate_MVC(testing_graphs[i])))
        A2 = nx.adjacency_matrix(testing_graphs[i]).toarray()
        v1 = A1.flatten()
        v2 = A2.flatten()
        cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
        print(f"modify_count: {modify_count}, cos_sim: {cos_sim}")
    label_presever = 0
    for i in range(len(testing_graphs)):
        if (mvc_ans[i] == opt[i]):
            label_presever += 1
    print(f"label_presever: {label_presever}") 
    #直接random 200條邊來修改的結果，1000個testing中大概是390個左右label preserve
    #直接random 200條邊來修改的結果，200個validation中大概是76個左右label preserve，cosine similarity大概是0.8左右
    #直接random 400條邊來修改的結果，200個validation中大概是33個左右label preserve，cosine similarity大概是0.6左右
