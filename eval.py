import math

import dgl
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os

from models.paper_recommendation import PrTextTAGC
from utils.loader import load_graph
from utils.metrics import Metrics
from utils.path import get_model_path, get_checkpoint_path

torch.manual_seed(0)
np.random.seed(0)
dgl.seed(0)


parser = argparse.ArgumentParser('Paper recommendation model evaluation')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. aan or dblp)',
                    default='dblp')
parser.add_argument('--prefix', type=str, default='pr-text-tagc',
                    help='Prefix to the loaded model and checkpoints')
parser.add_argument('--test-size', type=int, default=0,
                    help='Test set size for evaluation. If less than or equal to 0, use all test sets')
parser.add_argument('--n_epoch', type=int, default=50, help='The epochs of checkpoint')
parser.add_argument('--bs', type=int, default=44019   , help='每一轮给一篇测试论文选的候选论文数量')
# 这个参数主要是用来降低内存的，一次性计算所有候选论文的评分又可能会爆内存。内存够的，aan数据集里你就设为12390，评估会快很多
parser.add_argument('--undirected', action='store_true',
                    help='Determine whether to use an undirected graph ')
parser.add_argument('--sampler', type=str, default="full", choices=['', 'fixed', 'full'], #PR-TextTAGC测试时默认使用全采用 结果会好很多
                    help='Type of Neighbors sampler',)
parser.add_argument('--n_degree', type=int, default=15, help='Number of neighbors to sample')
parser.add_argument('--sample_dir', default="in", type=str, choices=['in', 'out'],
                    help="Determines whether the neighbor sampler sample inbound or outbound edges.")

parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--add_self_loop', action='store_true', help='Determine whether to add self loop in graph')
parser.add_argument('--have_gcn', action='store_true', help='如果有gcn的模型,则执行set_sampler')

args = parser.parse_args()

# set global params
GPU = args.gpu
BATCH_SIZE = args.bs
DATA = args.data
NUM_NEIGHBORS = args.n_degree
TEST_SIZE = args.test_size
HAVE_GCN = args.have_gcn


def get_model(device):
    """
    :param device: 模型训练时所在的设备
    :return: 返回模型实例
    """
    model_path = get_model_path(args.prefix, args.data)
    model = torch.load(model_path, map_location=device)  # 加载模型
    checkpoint_epoch = args.n_epoch - 1  # checkpoint的epoch从0开始计数
    if checkpoint_epoch >= 0:
        # 当指定args.n_epoch时，则加载指定epoch下的模型
        checkpoint_path = get_checkpoint_path(args.prefix, args.data, checkpoint_epoch)
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"load checkpoint: {checkpoint_path}")
    return model


def set_sampler(model):
    if args.sampler == '':
        # 邻居采样器与训练时相同
        print("Sampler info: The neighbor sampler used during evaluation is the same as during training")
    elif args.sampler == 'fixed':
        # 采用固定数量的邻居采样
        model.sampler = dgl.dataloading.NeighborSampler([NUM_NEIGHBORS for _ in range(0, model.n_layers)],
                                                        edge_dir=args.sample_dir)
        print(f"Sampler info: The number of neighbors in each layer of the neighbor sampler is {str(NUM_NEIGHBORS)}")
    elif args.sampler == 'full':
        model.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.n_layers, edge_dir=args.sample_dir)  # 无gcn 则没有model.n_layers参数，也不需要
        print(f"Sampler info: MultiLayerFullNeighborSampler")


def build_edge_dict(graph, edge_ids):
    # 基于边列表构建一个字典数据结构，
    # 该字典的键为`edge_ids`对应边所包含的所有源节点(src)，字典的值为源节点所指向的所有目标节点(dst)列表
    edge_dict = {}
    src_nodes, dst_nodes = graph.find_edges(edge_ids)
    for src, dst in zip(src_nodes, dst_nodes):
        src, dst = src.item(), dst.item()  # 由于原始src与dst均为tensor项，因此使用item()转为数值类型
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)
    return edge_dict


def evaluate(model: PrTextTAGC, g, train_eids, test_eids, device, is_undirected):
    """
        评估论文推荐模型的推荐性能，包含的指标有：MAP, MRR and Recall@[25, 50, 75, 100]
    :param model:
    :param g:
    :param train_eids:
    :param test_eids:
    :return:
    """
    # 根据训练边id提取的训练子图，但其中relabel_nodes=False用于保留测试论文节点
    train_graph = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)  # 训练图包含了所有节点（训练节点与测试节点），但只保留训练边
    if is_undirected:
        # 将图转为无向图，用于采样入和出邻居
        print("----use undirected graph----")
        train_graph = dgl.to_bidirected(train_graph, copy_ndata=True)
    if args.add_self_loop:
        train_graph = dgl.add_self_loop(train_graph, edge_feat_names=[])
    test_set = build_edge_dict(g, test_eids)  # 测试集：键包含测试论文的节点id，值包含该论文所引用论文的节点id列表，即正样本列表
    test_nodes = list(test_set.keys())

    all_nodes = train_graph.nodes()
    excluded_nodes_set = set(test_nodes).union(set([0]))  # 除了测试论文外，还要排除无意义的0节点
    mask = torch.tensor([node.item() not in excluded_nodes_set for node in all_nodes], dtype=torch.bool)
    train_nodes = all_nodes[mask]

    num_train_nodes = train_graph.num_nodes() - len(test_nodes)  # 候选论文数量
    batch_size = min(BATCH_SIZE, num_train_nodes)
    batch_num = math.ceil(num_train_nodes / batch_size)   # 向上取整算出来批次大小

    metrics = Metrics()

    with torch.no_grad():
        model = model.eval()
        train_nodes_ndarray = train_nodes.cpu().numpy()
        if TEST_SIZE > 0:
            test_nodes = np.random.choice(test_nodes, TEST_SIZE, replace=False)

        # 第一个循环的是所有测试节点  第二个循环的是一个测试节点对应的全部候选节点
        for i, test_node in tqdm(enumerate(test_nodes), total=len(test_nodes), desc="eval model"):
            scores = []
            timestamp = g.nodes[test_node].data['year']
            for k in range(batch_num):
                start = k * batch_size
                end = min(start + batch_size, num_train_nodes)
                destination_nodes = train_nodes[start: end]
                source_nodes = torch.full((len(destination_nodes),), test_node)
                timestamps = timestamp.repeat(len(destination_nodes))
                pred_score = model.predict_edge_probabilities(train_graph, source_nodes, destination_nodes, device)
                scores.extend(pred_score.view(-1).tolist())  # 展开为一维再添加

            sorted_indices = np.argsort(scores)[::-1]  # 所有候选节点评分计算完毕后 对得分进行排序，以获取排名从高到低的候选节点索引
            ranked_recommendation = train_nodes_ndarray[sorted_indices]
            metrics.add(ranked_recommendation, test_set[test_node])  # 计算指标
        # end evaluate
    metrics.printf()  # 打印结果


def main():
    print(args)
    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    g = load_graph(args.data)
    train_eids = torch.nonzero(g.edata['train_mask']).squeeze()
    test_eids = torch.nonzero(g.edata['test_mask']).squeeze()

    # Get model
    model = get_model(device)
    if HAVE_GCN:
        set_sampler(model)
    model.to(device)

    evaluate(model, g, train_eids, test_eids, device, args.undirected)


if __name__ == '__main__':
    main()
