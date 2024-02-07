import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.utils import to_networkx
import random
from heapdict import heapdict
from node2vec import Node2Vec
import argparse
from utils import *

graph_embedding_size = 64
edge_sample_number = 100
edge_dict = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self,num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, graph_embedding_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 定义GAT模型
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_heads=8):
        super(GAT, self).__init__()
        self.gat1 = GATConv(num_features, 128, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(128 * num_heads, graph_embedding_size, heads=1, concat=False, dropout=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        return x

# 定義用來決定edge是否修改的MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
class MLPClassifier(nn.Module):  #最後用來判定graph的result是否有相同的MVC
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层
        self.fc2 = nn.Linear(128, 64)          # 第二层
        self.fc3 = nn.Linear(64, 1)           # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))        # 使用sigmoid确保输出在0到1之间
        return x

class Modify_edge(nn.Module):
    def __init__(self, num_features, graph_embedding_size, epoch = 100, lr = 0.0001, modified_edge = 30, device = 'cuda:0', GraphNumber = 50, Graphsize = 50, num_heads = 8):
        super(Modify_edge, self).__init__()
        self.gat = GAT(num_features=num_features, num_heads=num_heads)  # 根据需要调整头数
        self.mlp = MLP(input_size=2 * graph_embedding_size + 1)
        self.classifier = MLPClassifier(input_size=2 * graph_embedding_size)
        self.modified_edge = modified_edge
        self.epoch = epoch
        self.lr = lr
        self.GraphNumber = GraphNumber
        self.Graphsize = Graphsize
        
        
    def forward(self):
        self.modified_graphs = []
        self.edge_dict = {}
        self.whole_edge_set = set()
        self.dataset = []
        self.init_graph()
        for data in self.dataset:
            data = data.to(device)
            gat_output = self.gat(data)
            combined_edge_embeddings, cur_edge_set = self.generate_edge_embeddings(data, gat_output)
            G = to_networkx(data, to_undirected=True)
            modify_num = 0
            decisions = []
            for edge_embeddings in combined_edge_embeddings:
                probabilities = self.mlp(edge_embeddings).squeeze()
                bernoulli = torch.distributions.Bernoulli(probabilities)
                decisions.append(bernoulli.sample().item())

            for i, decision in enumerate(decisions):
                if decision == 1:
                    modify_num += 1
                    edge = cur_edge_set[i]
                    if (i < edge_sample_number):
                        G.remove_edge(edge[0], edge[1])
                    else:
                        G.add_edge(edge[0], edge[1])
            print(f"modify_num: {modify_num}")
            self.modified_graphs.append(G)
        
        self.modified_dataset = []  #type pyg
        for G in self.modified_graphs:
            # 从 NetworkX 图创建边索引
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            
            # 使用单位矩阵作为节点特征
            vec = Node2Vec(G, dimensions=50, walk_length=15, num_walks=10, workers=4, quiet=True)
            InitNodeEmb = vec.fit(window=10, min_count=1, batch_words=4)
            embeddings = InitNodeEmb.wv
            x = torch.tensor(embeddings.vectors, dtype=torch.float32)
            # x = torch.eye(G.number_of_nodes())
            
            # 创建 Data 对象
            data = Data(x=x, edge_index=edge_index)
            self.modified_dataset.append(data)  #networkx
            
        modified_embeddings = []
        for data in self.modified_dataset:
            data = data.to(device)
            embedding = self.gat(data)
            modified_embeddings.append(embedding)
            
        original_embeddings = []
        for data in self.dataset:
            data = data.to(device)
            embedding = self.gat(data)
            original_embeddings.append(embedding)
            
        modified_graph_embeddings = self.get_graph_embedding(modified_embeddings)
        original_graph_embeddings = self.get_graph_embedding(original_embeddings)
        cos = nn.CosineSimilarity(dim=1)
        self.cosine_similarities = cos(modified_graph_embeddings, original_graph_embeddings).mean()
        
        labels = []
        MVC_diff = 0
        for mod_graph, orig_graph in zip(self.modified_graphs, self.dataset):
            mod_mvc = len(self.calculate_MVC(mod_graph))
            orig_mvc = len(self.calculate_MVC(to_networkx(orig_graph, to_undirected=True)))
            # print(f"mod_mvc: {mod_mvc}, ori_mvc: {orig_mvc}")
            MVC_diff += abs(mod_mvc - orig_mvc)
            label = 1 if mod_mvc == orig_mvc else 0
            labels.append(label)
        print(f"label presreved: {labels.count(1)}")
        combined_embeddings = [torch.cat((mod_emb, orig_emb)) for mod_emb, orig_emb in zip(modified_graph_embeddings, original_graph_embeddings)]
        # 将嵌入和标签转换为张量
        combined_embeddings_tensor = torch.stack(combined_embeddings)
        # combined_embeddings_tensor shape : torch.Size([50, 2*graph embedding]) 兩張graph的嵌入拼接起來
        self.labels_tensor = torch.tensor(labels).to(device)
        # labels_tensor shape : torch.Size([50]) 也就是50個graph的label
        self.preserve_predict = self.classifier(combined_embeddings_tensor).squeeze()
        # preserve_predict shape: torch.Size([50])也就是50個graph預測的label
        return self.cosine_similarities, self.preserve_predict, self.labels_tensor, MVC_diff/self.GraphNumber
        
        
    def init_graph(self):
        """construct or load training graph and use Node2vec to get node embedding"""
        self.train_graphs = pickle_load("/workspace/Synthetic_graph/Training_graph_50.pkl")
        self.create_edge_dict(self.Graphsize)
        for i in range(self.GraphNumber):
            # p = random.uniform(graph_density[0], graph_density[1])
            # G = nx.erdos_renyi_graph(graph_size, p)
            G = self.train_graphs[i]
            adj_matrix = nx.adjacency_matrix(G)
            adj_matrix = coo_matrix(adj_matrix)

            row = torch.from_numpy(adj_matrix.row.astype(np.int64))
            col = torch.from_numpy(adj_matrix.col.astype(np.int64))
            edge_index = torch.stack([row, col], dim=0)
            vec = Node2Vec(G, dimensions=50, walk_length=15, num_walks=10, workers=4, quiet=True)
            InitNodeEmb = vec.fit(window=10, min_count=1, batch_words=4)
            embeddings = InitNodeEmb.wv
            x = torch.tensor(embeddings.vectors, dtype=torch.float32)
            # x = torch.eye(G.number_of_nodes())  # 节点特征

            data = Data(x=x, edge_index=edge_index)
            self.dataset.append(data)
            
    def create_edge_dict(self,graph_size):
        """mapping edge to index"""
        index = 0
        for i in range(graph_size - 1):
            for j in range(i + 1, graph_size):
                self.whole_edge_set.add((i, j))
                self.edge_dict[(i, j)] = index
                index += 1
        
    def calculate_MVC(self,graph, UB=9999999, C=set()):
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
            C1 = self.calculate_MVC(graph_1, UB, C1)

        # C2 分支：只選擇該節點
        C2 = C.copy()
        C2.add(v)
        graph_2 = graph.copy()
        graph_2.remove_node(v)
        if len(C2) < UB:
            C2 = self.calculate_MVC(graph_2, min(UB, len(C1)), C2)

        return min(C1, C2, key=len)
    
    def get_graph_embedding(self,embeddings):
        """average all node embeddings to get graph embedding"""
        graph_embeddings = []
        for embedding in embeddings:
            graph_embedding = embedding.mean(dim=0)  # 对所有节点嵌入求平均
            graph_embeddings.append(graph_embedding)
        return torch.stack(graph_embeddings)
    
    def generate_edge_embeddings(self,data, embedding):
        """generate and sample edge embeddings for training 需要修改"""
        data= to_networkx(data, to_undirected=True)
        edge_set = set(data.edges()) 
        none_edge_set = self.whole_edge_set - edge_set  
        # print(f"edge_set: {len(edge_set)}, none_edge_set: {len(none_edge_set)}, whole_edge_set: {len(self.whole_edge_set)}")
        # select_edge_set = random.sample(edge_set, edge_sample_number)
        # select_none_edge_set = random.sample(none_edge_set, edge_sample_number)
        combined_embeddings = []
        # for u,v in select_edge_set:
        for u,v in edge_set:
            node1_emb = embedding[u]
            node2_emb = embedding[v]
            edge_emb = torch.cat([node1_emb, node2_emb, torch.tensor([1.0]).to(device)])
            combined_embeddings.append(edge_emb)
        # for u,v in select_none_edge_set:
        for u,v in none_edge_set:   
            node1_emb = embedding[u]
            node2_emb = embedding[v]
            none_edge_emb = torch.cat([node1_emb, node2_emb, torch.tensor([0.0]).to(device)])
            combined_embeddings.append(none_edge_emb)
        # cur_edge_set = select_edge_set + select_none_edge_set
        cur_edge_set = list(edge_set) + list(none_edge_set)
        return combined_embeddings, cur_edge_set
    
if __name__ == '__main__':
    mymodel = Modify_edge(num_features=50, graph_embedding_size=graph_embedding_size, epoch=100, lr=0.0001, modified_edge=edge_sample_number, device=device, GraphNumber=50, Graphsize=50, num_heads=8)
    mymodel = mymodel.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.0001)
    diff_weight = 2
    for epoch in range(100):
        mymodel.train()
        similarity_loss , preserve_predict, labels_tensor, difference_loss = mymodel()
        classifier_loss = criterion(preserve_predict, labels_tensor.float())
        loss = classifier_loss + similarity_loss + difference_loss * diff_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss}, similarity_loss: {similarity_loss}, difference_loss: {difference_loss}")