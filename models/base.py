from torch import nn

from models.utils import merge_time_encoding
from modules.predictor import MLPsPredictor, AttnPredictor
from modules.time_encode import TimeEncode


class PrBase(nn.Module):
    """无图嵌入模块以及时序编码模块的推荐模型"""
    def __init__(self, raw_feats, pred_hidden_feats, dropout=0.1, predictor="mlp", predictor_heads=1):
        super().__init__()

        self.raw_feats = raw_feats  # 初始特征向量维度
        self.dropout = dropout
        self.predictor = predictor
        self.pred_hidden_feats = pred_hidden_feats  # 预测模块的隐藏层维度
        self.predictor_heads = predictor_heads

        # modules
        self._init_module_size()
        self.affinity_score = self._select_predictor()

    def _init_module_size(self):
        #
        self.in_feats = self.raw_feats
        self.src_output_feats = self.raw_feats  # src节点输出特征向量(即_get_src_embedding)的维度
        self.dst_output_feats = self.raw_feats  # dst节点输出特征向量(即_get_dst_embedding)的维度

    def _select_predictor(self):
        if self.predictor == "mlp":
            return MLPsPredictor(self.src_output_feats, self.dst_output_feats, self.pred_hidden_feats)
        elif self.predictor == "attn":
            return AttnPredictor(self.src_output_feats, self.dst_output_feats, self.pred_hidden_feats,
                                 self.predictor_heads, self.dropout)
        else:
            raise ValueError(f"Invalid merge type: {self.predictor}. "
                             f"Only 'mlp' and 'attn' are supported.")

    def forward(self, positive_graph, negative_graph, blocks, x):
        src_embeddings = self._get_src_embedding(positive_graph, positive_graph.nodes())
        dst_embeddings = src_embeddings  # 训练时，src_embeddings与dst_embeddings均包含图中所有节点的特征，因此相同

        pos_score = self.affinity_score(positive_graph, src_embeddings, dst_embeddings)
        neg_score = self.affinity_score(negative_graph, src_embeddings, dst_embeddings)

        return pos_score, neg_score

    def predict_edge_probabilities(self, graph, src_nodes, dst_nodes, device):
        graph = graph.to(device)
        src_nodes = src_nodes.to(device)
        dst_nodes = dst_nodes.to(device)
        src_embeddings = self._get_src_embedding(graph, src_nodes).to(device)
        dst_embeddings = self._get_dst_embedding(graph, dst_nodes).to(device)

        score = self.affinity_score.predict(src_embeddings, dst_embeddings)
        return score

    def _get_src_embedding(self, graph, src_nodes):
        return graph.ndata['feat'][src_nodes]

    def _get_dst_embedding(self, graph, dst_nodes):
        return self._get_src_embedding(graph, dst_nodes)


class PrTAText(PrBase):
    def __init__(self, raw_feats, time_feats, pred_hidden_feats, dropout=0.1,
                 time_merge_type="add", predictor="mlp", predictor_heads=1):
        self.time_merge_type = time_merge_type
        self.time_feats = time_feats
        super().__init__(raw_feats, pred_hidden_feats, dropout, predictor, predictor_heads)

        # new modules
        self.time_encoder = TimeEncode(dimension=self.time_feats)

    def _init_module_size(self):
        super()._init_module_size()
        if self.time_merge_type == "concat":
            self.src_output_feats += self.time_feats
            self.dst_output_feats += self.time_feats

    def _get_src_embedding(self, graph, src_nodes):
        text_embedding = graph.ndata['feat'][src_nodes]
        timestamps = graph.ndata['year'][src_nodes]
        return merge_time_encoding(text_embedding, timestamps, self.time_encoder, self.time_merge_type)

