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
parser = argparse.ArgumentParser()
parser.add_argument("--iteration", default= 100, type=int)
parser.add_argument("--modify_edge", default= 100, type=int)
parser.add_argument("--graph_embedding", default= 64, type=int)
parser.add_argument("--GraphNumber", default= 50, type=int)
parser.add_argument("--GraphSize", default= 50, type=int)
parser.add_argument("--weight", default= 1, type=int)
args = parser.parse_args()
graph_density = (0.25, 0.75)
graph_embedding_size = args.graph_embedding
edge_sample_number = args.modify_edge
edge_dict = {}
whole_edge_set = set()
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

# 计算图级嵌入：对所有节点的嵌入进行平均
def get_graph_embedding(embeddings):
    graph_embeddings = []
    for embedding in embeddings:
        graph_embedding = embedding.mean(dim=0)  # 对所有节点嵌入求平均
        graph_embeddings.append(graph_embedding)
    return torch.stack(graph_embeddings)

def create_edge_dict(graph_size):
    index = 0
    for i in range(graph_size - 1):
        for j in range(i + 1, graph_size):
            whole_edge_set.add((i, j))
            edge_dict[(i, j)] = index
            index += 1

def generate_edge_embeddings(dataset, embedding):
    data= to_networkx(dataset, to_undirected=True)
    edge_set = set(data.edges()) 
    none_edge_set = whole_edge_set - edge_set  
    select_edge_set = random.sample(edge_set, edge_sample_number)
    select_none_edge_set = random.sample(none_edge_set, edge_sample_number)
    combined_embeddings = []
    for u,v in select_edge_set:
        node1_emb = embedding[u]
        node2_emb = embedding[v]
        edge_emb = torch.cat([node1_emb, node2_emb, torch.tensor([1.0]).to(device)])
        combined_embeddings.append(edge_emb)
    for u,v in select_none_edge_set:
        node1_emb = embedding[u]
        node2_emb = embedding[v]
        none_edge_emb = torch.cat([node1_emb, node2_emb, torch.tensor([0.0]).to(device)])
        combined_embeddings.append(none_edge_emb)
    cur_edge_set = select_edge_set + select_none_edge_set
    return combined_embeddings, cur_edge_set

if __name__ == '__main__':
    dataset = []
    graph_size = args.GraphSize
    number_of_graphs = args.GraphNumber
    create_edge_dict(graph_size)
    train_graphs = pickle_load("/workspace/Synthetic_graph/Training_graph_50.pkl")
    for i in range(number_of_graphs):
        # p = random.uniform(graph_density[0], graph_density[1])
        # G = nx.erdos_renyi_graph(graph_size, p)
        G = train_graphs[i]
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
        dataset.append(data)
    # 实例化GAT模型
    model = GAT(num_features=graph_size)
    # model = GCN(num_features=graph_size)
    model.to(device)
    # 实例化MLP模型
    embedding_size = 2 * graph_embedding_size + 1
    mlp = MLP(input_size=embedding_size)
    mlp.to(device)
    classifier = MLPClassifier(2*graph_embedding_size)
    classifier.to(device)
    # 定义损失函数
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    cos = nn.CosineSimilarity(dim=1)

    # 假设一些训练超参数
    epochs = args.iteration
    learning_rate = 0.0001
    # 优化器包括GAT和MLP模型的参数
    optimizer = torch.optim.Adam(list(model.parameters()) + list(mlp.parameters()) + list(classifier.parameters()), lr=learning_rate)
    loss_log = []
    cosine_log = []
    classification_log = []
    MVC_diff_log = []
    label_preserved_log = []
    
    for epoch in range(epochs):
        total_loss = 0
        modified_graphs = []
        model.train()
        mlp.train()
        classifier.train()
        print(f"epoch: {epoch}")
        for data in dataset:
            optimizer.zero_grad()
            data = data.to(device)
            # GAT模型前向传播
            gat_output = model(data)

            # 获取每张图的边和非边嵌入
            combined_edge_embeddings, cur_edge_set = generate_edge_embeddings(data, gat_output)
            
            # 添加原始存在的边
            G = to_networkx(data, to_undirected=True)
            # G = nx.Graph()
            # G.add_nodes_from(range(data.num_nodes))
            # for edge in data.edge_index.t().numpy():
            #         G.add_edge(*edge)
                    
            modify_num = 0
            decisions = []
            for edge_embeddings in combined_edge_embeddings:
                probabilities = mlp(edge_embeddings).squeeze()
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
            modified_graphs.append(G)
        
        modified_dataset = []
        for G in modified_graphs:
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
            modified_dataset.append(data)  #networkx
            
        modified_embeddings = []
        for data in modified_dataset:
            data = data.to(device)
            embedding = model(data)
            modified_embeddings.append(embedding)
            
        original_embeddings = []
        for data in dataset:
            data = data.to(device)
            embedding = model(data)
            original_embeddings.append(embedding)
            
        modified_graph_embeddings = get_graph_embedding(modified_embeddings)
        original_graph_embeddings = get_graph_embedding(original_embeddings)
        # # 计算兩張圖的graph embedding的余弦相似度，用來作為similarity loss
        cos = nn.CosineSimilarity(dim=1)
        cosine_similarities = cos(modified_graph_embeddings, original_graph_embeddings).mean()
        # # cosine_similarities shape: torch.Size([50]) 50 張圖之間的相似度之後進行平均
        print(f"cosine_similarities: {cosine_similarities.item()}")
        cosine_log.append(cosine_similarities.item())
        # # classification loss
        labels = []
        MVC_diff = 0
        for mod_graph, orig_graph in zip(modified_graphs, dataset):
            mod_mvc = len(calculate_MVC(mod_graph))
            orig_mvc = len(calculate_MVC(to_networkx(orig_graph, to_undirected=True)))
            print(f"mod_mvc: {mod_mvc}, orig_mvc: {orig_mvc}")
            MVC_diff += abs(mod_mvc - orig_mvc)
            label = 1 if mod_mvc == orig_mvc else 0
            labels.append(label)
        print(f"label presreved: {labels.count(1)}")
        label_preserved_log.append(labels.count(1))
        combined_embeddings = [torch.cat((mod_emb, orig_emb)) for mod_emb, orig_emb in zip(modified_graph_embeddings, original_graph_embeddings)]
        # 将嵌入和标签转换为张量
        combined_embeddings_tensor = torch.stack(combined_embeddings)
        # combined_embeddings_tensor shape : torch.Size([50, 2*graph embedding]) 兩張graph的嵌入拼接起來
        labels_tensor = torch.tensor(labels).to(device)
        # labels_tensor shape : torch.Size([50]) 也就是50個graph的label
        preserve_predict = classifier(combined_embeddings_tensor).squeeze()
        # preserve_predict shape: torch.Size([50])也就是50個graph預測的label
        classification_loss = criterion(preserve_predict.float(), labels_tensor.float())
        print(f"classification_loss: {classification_loss.item()}")
        classification_log.append(classification_loss.item())
        loss = classification_loss + cosine_similarities + args.weight*(MVC_diff / number_of_graphs)
        print("MVC diff: ", MVC_diff / number_of_graphs)
        MVC_diff_log.append(MVC_diff / number_of_graphs)
        print(f"loss: {loss}")
        loss_log.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with open("train_log_4.txt", "w") as f:
        f.write("label_preserved_log: ")
        f.write(str(label_preserved_log))
        f.write("\n")
        f.write("MVC_diff_log: ")
        f.write(str(MVC_diff_log))
        f.write("\n")
        f.write("loss_log: ")
        f.write(str(loss_log))
        f.write("\n")
        f.write("cosine_log: ")
        f.write(str(cosine_log))
        f.write("\n")
        f.write("classification_log: ")
        f.write(str(classification_log))
        f.write("\n")
        
"""
python Test.py --iteration 100 --modify_edge 30 --graph_embedding 64 --GraphNumber 50 --GraphSize 50 --weight 2
"""