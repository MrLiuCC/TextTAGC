from torch import nn
from modules.graph_conv import SAGE, GAT
from modules.predictor import MLPsPredictor, AttnPredictor


class PrGC(nn.Module):
    def __init__(self, raw_feats, hidden_dims, pred_hidden_feats, sampler,
                 num_layers=2, num_heads=1, dropout=0.1,
                 gcn_module="sage", predictor="mlp",
                 predictor_heads=1):
        super(PrGC, self).__init__()

        self.raw_feats = raw_feats  # 初始特征向量维度

        self.gcn_module = gcn_module.lower()
        self.gcn_hidden_dims = hidden_dims  # 图卷积模块的隐藏层维度
        self.gcn_output_feats = hidden_dims[len(hidden_dims)-1]  # 图卷积模块的输出维度，即隐藏层的最后一项
        self.pred_hidden_feats = pred_hidden_feats  # 预测模块的隐藏层维度
        self.predictor = predictor
        self.predictor_heads = predictor_heads

        self.n_layers = num_layers
        self.n_heads = num_heads
        self.dropout = dropout
        self.sampler = sampler

        # modules  使用了类的内部函数 在初始化部分直接调用
        self._init_module_size()
        self.graph_embedding_layer = self._select_gcn_module()
        self.affinity_score = self._select_predictor()

    def _init_module_size(self):
        # 初始化模型各个神经网络模块的参数规模
        self.in_feats = self.raw_feats
        self.src_output_feats = self.gcn_output_feats  # src节点输出特征向量(即_get_src_embedding)的维度
        self.dst_output_feats = self.gcn_output_feats  # dst节点输出特征向量(即_get_dst_embedding)的维度

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
        dst_embeddings = self._get_dst_embedding(blocks, x)
        src_embeddings = dst_embeddings  # 训练时，src_embeddings与dst_embeddings均为图中所有节点的嵌入向量

        pos_score = self.affinity_score(positive_graph, src_embeddings, dst_embeddings)
        neg_score = self.affinity_score(negative_graph, src_embeddings, dst_embeddings)

        return pos_score, neg_score

    def predict_edge_probabilities(self, graph, src_nodes, dst_nodes, device):
        # 评估时调用，用于预测推荐评分
        _, _, src_blocks = self.sampler.sample(graph, src_nodes)
        src_blocks = [b.to(device) for b in src_blocks]
        src_embeddings = self._get_src_embedding(src_blocks, src_blocks[0].srcdata['feat'])

        _, _, dst_blocks = self.sampler.sample(graph, dst_nodes)
        dst_blocks = [b.to(device) for b in dst_blocks]
        dst_embeddings = self._get_dst_embedding(dst_blocks, dst_blocks[0].srcdata['feat'])

        score = self.affinity_score.predict(src_embeddings, dst_embeddings)
        return score

    def _get_dst_embedding(self, blocks, x):
        dst_embedding = self.graph_embedding_layer(blocks, x)
        return dst_embedding

    def _get_src_embedding(self, blocks, x):
        return self._get_dst_embedding(blocks, x)
