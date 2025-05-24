import argparse
import time

import dgl
import os
import numpy as np
import torch
import logging

from models.base import PrBase, PrTAText
from models.paper_recommendation import PrTextTAGC, PrTextGC
from models.pr_gc import PrGC
from utils.loader import load_graph
from utils.path import get_checkpoint_path, get_model_path
from utils.sampler import select_negative_sampler

torch.manual_seed(0)
np.random.seed(0)
dgl.seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser('Train paper recommendation model')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. aan or dblp)', default='dblp')
    parser.add_argument('--prefix', type=str, default='pr-text-tagc', help='Prefix to name the model and checkpoints')
    # model parameter
    parser.add_argument('--model', type=str, default='pr-text-tagc',
                        help='Model name (eg. pr-base ,pr-ta-text, pr-gc, pr-text-tagc ,pr-text-gc)')
    parser.add_argument('--bs', type=int, default=1024, help='Batch_size')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--full_sample', action='store_true',
                        help='Determine whether to sample all neighbors for train')
    parser.add_argument('--negative_sampler', type=str, default="uniform", choices=['uniform', 'time'],
                        help='Negative sampler')
    parser.add_argument('--n_negative', type=int, default=1, help='Number of negative samples')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epochs')
    # 训练总轮次数 一般为800、900或1000
    parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of GAT heads. Valid only when gcn_module=gat')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')

    # parser.add_argument('--reverse', action='store_true',
    #                     help="Determines whether the neighbor sampler sample outbound edges.")

    parser.add_argument('--undirected', action='store_true',
                        help='Determine whether to use an undirected graph')
    parser.add_argument('--gcn_module', default="sage", type=str, choices=['sage', 'gat'],
                        help="Determine the graph convolutional network module used")
    parser.add_argument('--predictor', default="mlp", type=str, choices=['mlp', 'attn'],
                        help="Determine the predictor module used")
    parser.add_argument('--gcn_merge_type', default="concat", type=str, choices=['add', 'concat', 'none'],
                        help="GCN embedding combination method")
    parser.add_argument('--time_merge_type', default="concat", type=str, choices=['add', 'concat', 'none'],
                        help="Time encoding combination method")

    # embedding dim
    parser.add_argument('--node_dim', type=int, default=768, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=768, help='Dimensions of the time encoding embedding')
    parser.add_argument('--pred_dim', type=int, default=768, help='Dimensions of hidden layer in the predictor module')
    parser.add_argument('--pred_head', type=int, default=1,
                        help='Number of predictor attn heads. Valid only when predictor=attn')
    parser.add_argument('--add_self_loop', action='store_true',
                        help='Determine whether to add self loop in graph')

    # others
    parser.add_argument('--interval', type=int, default=20, help='The interval to save checkpoint')

    args = parser.parse_args()
    return args


def compute_loss(pos_score, neg_score):
    n = pos_score.shape[0]
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()


def select_model(args, sampler):
    logging.info(f"current model: {args.model}")
    if args.model == "pr-base":
        return PrBase(args.node_dim, args.pred_dim, dropout=args.dropout,
                      predictor=args.predictor, predictor_heads=args.pred_head)
    elif args.model == "pr-ta-text":
        return PrTAText(args.node_dim, args.time_dim, args.pred_dim, dropout=args.dropout,
                        time_merge_type=args.time_merge_type, predictor=args.predictor, predictor_heads=args.pred_head)
    elif args.model == "pr-gc":
        return PrGC(args.node_dim, [args.node_dim for _ in range(0, args.n_layer)], args.pred_dim, sampler,
                    args.n_layer,  num_heads=args.n_head, dropout=args.dropout, gcn_module=args.gcn_module,
                    predictor=args.predictor, predictor_heads=args.pred_head)
    elif args.model == "pr-text-gc":
        return PrTextGC(args.node_dim, [args.node_dim for _ in range(0, args.n_layer)], args.pred_dim, sampler,
                        args.n_layer,  num_heads=args.n_head, dropout=args.dropout,
                        gcn_module=args.gcn_module, gcn_merge_type=args.gcn_merge_type, predictor=args.predictor,
                        predictor_heads=args.pred_head)
    elif args.model == "pr-text-tagc":
        return PrTextTAGC(args.node_dim, args.time_dim, [args.node_dim for _ in range(0, args.n_layer)], args.pred_dim,
                          sampler, args.n_layer, num_heads=args.n_head, dropout=args.dropout,
                          gcn_module=args.gcn_module, gcn_merge_type=args.gcn_merge_type,
                          time_merge_type=args.time_merge_type, predictor=args.predictor,
                          predictor_heads=args.pred_head)
    else:
        raise ValueError(f"Invalid merge type: {args.model}. "
                         f"Only ['pr-base', 'pr-ta-text', 'pr-gc', 'pr-text-gc','pr-text-tagc' are supported.")


def build_undirected_train_graph(g: dgl.DGLGraph):
    train_eids = torch.nonzero(g.edata['train_mask']).squeeze()
    bidirected_g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=False)  # 新增的反向边都位于edges()的末尾
    bidirected_g.edata['train_mask'] = torch.full((bidirected_g.number_of_edges(),), False, dtype=torch.bool)
    bidirected_g.edata['train_mask'][train_eids] = True
    bidirected_g = dgl.to_simple(bidirected_g, return_counts=None, copy_ndata=True, copy_edata=True)
    return bidirected_g


def main():
    args = parse_arguments()
    print(args)
    NUM_NEGATIVE = args.n_negative

    # set logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("cur device: ", device)

    # load data
    g = load_graph(args.data)
    train_nids = torch.nonzero(g.ndata['train_mask']).squeeze()
    test_eids = torch.nonzero(g.edata['test_mask']).squeeze()

    train_graph = dgl.node_subgraph(g, train_nids)
    if args.undirected:
        train_graph = build_undirected_train_graph(train_graph)
    if args.add_self_loop:
        train_graph = dgl.add_self_loop(train_graph, edge_feat_names=[])

    train_graph.to(device)
    train_seeds = torch.nonzero(train_graph.edata['train_mask']).squeeze()
    print("total graph: ", g)
    print("train graph:", train_graph)

    if args.model == "pr-base" or args.model == "pr-ta-text":
        args.n_degree = 0

    if args.full_sample:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layer)
    else:
        sampler = dgl.dataloading.NeighborSampler([args.n_degree for _ in range(0, args.n_layer)])

    negative_sampler = select_negative_sampler(args.negative_sampler, NUM_NEGATIVE)
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)
    train_dataloader = dgl.dataloading.DataLoader(
        train_graph, train_seeds, edge_sampler,
        batch_size=args.bs,
        device=device,
        shuffle=False,
        drop_last=False,
        num_workers=1)

    model = select_model(args, sampler)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MarginRankingLoss(margin=1.0)

    # start train
    for epoch in range(args.n_epoch):
        epoch_start = time.time()

        model = model.train()
        m_loss = []
        logging.info('start {} epoch'.format(epoch))
        for input_nodes, positive_graph, negative_graph, blocks in train_dataloader:
            optimizer.zero_grad()
            blocks = [b.to(device) for b in blocks]
            positive_graph = positive_graph.to(device)
            negative_graph = negative_graph.to(device)
            input_features = blocks[0].srcdata['feat']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            pos_score = pos_score.squeeze().repeat_interleave(NUM_NEGATIVE)  # 使得pos_score与neg_score的shape保持一致
            neg_score = neg_score.squeeze()
            ys = torch.ones(len(pos_score), dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_score, neg_score, ys)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
        # end one epoch and log
        logging.info('epoch: {} took {:.2f}s'.format(epoch, time.time() - epoch_start))  # 记录一轮epoch花费时间
        logging.info('Epoch mean loss: {}'.format(np.mean(m_loss)))  # 记录训练损失
        # save checkpoint
        if (epoch + 1) % args.interval == 0:
            torch.save(model.state_dict(), get_checkpoint_path(args.prefix, args.data, epoch))  # save checkpoint

    # end train
    # save model
    logging.info('Saving TGN model')
    torch.save(model, get_model_path(args.prefix, args.data))


if __name__ == '__main__':
    main()
