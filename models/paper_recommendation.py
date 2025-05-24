from abc import ABC, abstractmethod

import numpy as np
import torch
import dgl
from torch import nn

from models.utils import merge_features
from modules.graph_conv import SAGE, GAT
from modules.time_encode import TimeEncode
from modules.predictor import MLPsPredictor, AttnPredictor


# torch.manual_seed(0)
# np.random.seed(0)
# dgl.seed(0)


# 定义了一个抽象基类、必须实现的抽象方法
class IPaperRecommendation(ABC):
    @abstractmethod
    def predict_edge_probabilities(self, src_nodes, dst_nodes, edge_times, n_neighbors=20):
        pass

    @abstractmethod
    def embedding_nodes(self, nodes):
        pass

    @abstractmethod
    def predict(self):
        pass


class PrTextGC(nn.Module):
    def __init__(self, raw_feats, hidden_dims, pred_hidden_feats, sampler,
                 num_layers=2, num_heads=1, dropout=0.1,
                 gcn_module="sage", gcn_merge_type="concat", predictor="mlp",
                 predictor_heads=1):
        super(PrTextGC, self).__init__()

        self.raw_feats = raw_feats  # 初始特征向量维度

        self.gcn_module = gcn_module.lower()
        self.gcn_hidden_dims = hidden_dims  # 图卷积模块的隐藏层维度
        self.gcn_output_feats = hidden_dims[len(hidden_dims) - 1]  # 图卷积模块的输出维度，即隐藏层的最后一项
        self.pred_hidden_feats = pred_hidden_feats  # 预测模块的隐藏层维度
        self.predictor = predictor
        self.predictor_heads = predictor_heads

        self.n_layers = num_layers
        self.n_heads = num_heads
        self.dropout = dropout
        self.sampler = sampler
        self.gcn_merge_type = gcn_merge_type.lower()  # 图嵌入向量与其他特征向量的结合方式

        # modules
        self._init_module_size()
        self.graph_embedding_layer = self._select_gcn_module()
        self.affinity_score = self._select_predictor()

    def _init_module_size(self):
        self.in_feats = self.raw_feats
        self.src_output_feats = self.raw_feats
        self.dst_output_feats = self.gcn_output_feats
        if self.gcn_merge_type == "concat":
            self.dst_output_feats += self.raw_feats

    def _select_gcn_module(self):
        if self.gcn_module == "sage":
            return SAGE(self.in_feats, self.gcn_hidden_dims, self.n_layers, dropout=self.dropout)
        elif self.gcn_module == "gat":
            return GAT(self.in_feats, self.gcn_hidden_dims, self.n_layers, num_heads=self.n_heads, dropout=self.dropout)
        else:
            # 不支持的图卷积类型
            raise ValueError(f"Invalid merge type: {self.gcn_module}. "
                             f"Only 'sage' and 'gat' are supported.")

    def _select_predictor(self):
        if self.predictor == "mlp":
            return MLPsPredictor(self.src_output_feats, self.dst_output_feats, self.pred_hidden_feats)
        elif self.predictor == "attn":
            return AttnPredictor(self.src_output_feats, self.dst_output_feats, self.pred_hidden_feats,
                                 self.predictor_heads, self.dropout)
        else:
            # 不支持的预测器类型
            raise ValueError(f"Invalid merge type: {self.predictor}. "
                             f"Only 'mlp' and 'attn' are supported.")

    def forward(self, positive_graph, negative_graph, blocks, x):
        src_embeddings = self._get_src_embedding(blocks)
        dst_embeddings = self._get_dst_embedding(blocks, x)

        pos_score = self.affinity_score(positive_graph, src_embeddings, dst_embeddings)
        neg_score = self.affinity_score(negative_graph, src_embeddings, dst_embeddings)

        return pos_score, neg_score

    def predict_edge_probabilities(self, graph, src_nodes, dst_nodes, device):
        # 评估时调用，用于预测推荐评分
        _, _, src_blocks = self.sampler.sample(graph, src_nodes)
        src_blocks = [b.to(device) for b in src_blocks]
        src_embeddings = self._get_src_embedding(src_blocks)

        _, _, dst_blocks = self.sampler.sample(graph, dst_nodes)
        dst_blocks = [b.to(device) for b in dst_blocks]
        dst_embeddings = self._get_dst_embedding(dst_blocks, dst_blocks[0].srcdata['feat'])

        score = self.affinity_score.predict(src_embeddings, dst_embeddings)
        return score

    def _get_dst_embedding(self, blocks, x):
        graph_embedding = self.graph_embedding_layer(blocks, x)
        text_embedding = blocks[len(blocks) - 1].dstdata['feat']
        dst_embedding = merge_features(graph_embedding, text_embedding, self.gcn_merge_type)
        return dst_embedding

    def _get_src_embedding(self, blocks):
        return blocks[len(blocks) - 1].dstdata['feat']


class PrTextTAGC(PrTextGC):
    def __init__(self, raw_feats, time_feats, hidden_dims, pred_hidden_feats, sampler,
                 num_layers=2, num_heads=1, dropout=0.1,
                 gcn_module="sage", gcn_merge_type="concat", time_merge_type="add", predictor="mlp",
                 predictor_heads=1):
        self.time_feats = time_feats
        self.time_merge_type = time_merge_type.lower()

        super(PrTextTAGC, self).__init__(raw_feats, hidden_dims, pred_hidden_feats, sampler,
                                         num_layers, num_heads, dropout,
                                         gcn_module, gcn_merge_type, predictor, predictor_heads)

        # new modules
        self.time_encoder = TimeEncode(dimension=self.time_feats)

    def _init_module_size(self):
        super()._init_module_size()
        self.in_feats += self.time_feats
        if self.time_merge_type == "concat":
            self.src_output_feats += self.time_feats
            self.dst_output_feats += self.time_feats

    def _get_dst_embedding(self, blocks, x):
        timestamps = blocks[0].srcdata['year']
        # 卷积过程中每个节点的时序编码固定使用concat连接
        x = self._merge_time_encoding(x, timestamps, time_combination="concat")
        graph_embedding = self.graph_embedding_layer(blocks, x)
        # 融合文本编码与时序编码
        text_embedding = blocks[len(blocks) - 1].dstdata['feat']
        dst_timestamps = blocks[len(blocks) - 1].dstdata['year']
        text_time_features = self._merge_time_encoding(text_embedding, dst_timestamps, self.time_merge_type)
        dst_embedding = merge_features(graph_embedding, text_time_features, self.gcn_merge_type)
        return dst_embedding

    def _get_src_embedding(self, blocks):
        text_embedding = blocks[len(blocks) - 1].dstdata['feat']
        timestamps = blocks[len(blocks) - 1].dstdata['year']
        return self._merge_time_encoding(text_embedding, timestamps, self.time_merge_type)

    def _merge_time_encoding(self, features, timestamps, time_combination: str):
        # 将特征向量与其时序编码向量结合，输出新的特征向量
        time_combination = time_combination.lower()
        assert timestamps.size()[0] == features.size()[0], \
            (f"The first dimensions of features[{str(features.size()[0])}] and timestamps[{str(timestamps.size()[0])}] "
             f"don't match")
        time_encoding = self.time_encoder(timestamps.float())
        # 将特征向量与时序编码结合
        combine_feats = merge_features(features, time_encoding, time_combination)
        return combine_feats
