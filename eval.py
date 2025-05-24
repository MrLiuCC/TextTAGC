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
parser.add_argument('--undirected', action='store_true',
                    help='Determine whether to use an undirected graph ')
parser.add_argument('--sampler', type=str, default="full", choices=['', 'fixed', 'full'],
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
    model = torch.load(model_path, map_location=device)
    checkpoint_epoch = args.n_epoch - 1
    if checkpoint_epoch >= 0:
        checkpoint_path = get_checkpoint_path(args.prefix, args.data, checkpoint_epoch)
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"load checkpoint: {checkpoint_path}")
    return model


def set_sampler(model):
    if args.sampler == '':
        print("Sampler info: The neighbor sampler used during evaluation is the same as during training")
    elif args.sampler == 'fixed':
        model.sampler = dgl.dataloading.NeighborSampler([NUM_NEIGHBORS for _ in range(0, model.n_layers)],
                                                        edge_dir=args.sample_dir)
        print(f"Sampler info: The number of neighbors in each layer of the neighbor sampler is {str(NUM_NEIGHBORS)}")
    elif args.sampler == 'full':
        model.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.n_layers, edge_dir=args.sample_dir)  # 无gcn 则没有model.n_layers参数，也不需要
        print(f"Sampler info: MultiLayerFullNeighborSampler")


def build_edge_dict(graph, edge_ids):
    edge_dict = {}
    src_nodes, dst_nodes = graph.find_edges(edge_ids)
    for src, dst in zip(src_nodes, dst_nodes):
        src, dst = src.item(), dst.item()
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)
    return edge_dict


def evaluate(model: PrTextTAGC, g, train_eids, test_eids, device, is_undirected):
    """
    MAP, MRR and Recall@[25, 50, 75, 100]
    :param model:
    :param g:
    :param train_eids:
    :param test_eids:
    :return:
    """
    # 根据训练边id提取的训练子图，但其中relabel_nodes=False用于保留测试论文节点
    train_graph = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)
    if is_undirected:
        print("----use undirected graph----")
        train_graph = dgl.to_bidirected(train_graph, copy_ndata=True)
    if args.add_self_loop:
        train_graph = dgl.add_self_loop(train_graph, edge_feat_names=[])
    test_set = build_edge_dict(g, test_eids)
    test_nodes = list(test_set.keys())

    all_nodes = train_graph.nodes()
    excluded_nodes_set = set(test_nodes).union(set([0]))
    mask = torch.tensor([node.item() not in excluded_nodes_set for node in all_nodes], dtype=torch.bool)
    train_nodes = all_nodes[mask]

    num_train_nodes = train_graph.num_nodes() - len(test_nodes)
    batch_size = min(BATCH_SIZE, num_train_nodes)
    batch_num = math.ceil(num_train_nodes / batch_size)

    metrics = Metrics()

    with torch.no_grad():
        model = model.eval()
        train_nodes_ndarray = train_nodes.cpu().numpy()
        if TEST_SIZE > 0:
            test_nodes = np.random.choice(test_nodes, TEST_SIZE, replace=False)


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
                scores.extend(pred_score.view(-1).tolist())

            sorted_indices = np.argsort(scores)[::-1]
            ranked_recommendation = train_nodes_ndarray[sorted_indices]
            metrics.add(ranked_recommendation, test_set[test_node])
        # end evaluate
    metrics.printf()


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
