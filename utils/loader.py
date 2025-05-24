import dgl
import os
import torch
import numpy as np


def load_graph(name, data_dir="./data"):
    dataset = dgl.data.CSVDataset(os.path.join(data_dir, name), force_reload=True)
    # print("加载dataset成功，本次加载的数据集为", name)
    features = torch.tensor(np.load(os.path.join(data_dir, name, "nodes_feat.npy")), dtype=torch.float32)
    #print("加载节点特征成功,本次加载的数据集为", name)

    graph = dataset[0]
    graph.ndata['feat'] = features
    return graph
