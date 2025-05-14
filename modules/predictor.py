import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class MLPsPredictor(nn.Module):
    def __init__(self, src_feats, dst_feats, hidden_dim):
        super().__init__()
        self.fc_src = nn.Linear(src_feats, hidden_dim)
        self.fc_dst = nn.Linear(dst_feats, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU()

        nn.init.xavier_normal_(self.fc_src.weight)
        nn.init.xavier_normal_(self.fc_dst.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, subgraph, src_feats, dst_feats):
        with subgraph.local_scope():
            out_src = self.fc_src(src_feats)
            out_dst = self.fc_dst(dst_feats)
            subgraph.srcdata.update({'out_src': out_src})
            subgraph.dstdata.update({'out_dst': out_dst})
            subgraph.apply_edges(dgl.function.u_add_v('out_src', 'out_dst', 'score'))
            # 线性映射为隐藏维度的向量，再相加，聚合特征，后使用激活函数，映射为得分
            score = self.fc_out(self.act(subgraph.edata['score']))
            return score

    def predict(self, src_feats, dst_feats):
        src_proj = self.fc_src(src_feats)
        dst_proj = self.fc_dst(dst_feats)
        score = self.fc_out(self.act(src_proj + dst_proj))
        return score


class AttnPredictor(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.Q_proj = nn.Linear(query_dim, hidden_dim * num_heads)
        self.K_proj = nn.Linear(key_dim, hidden_dim * num_heads)
        self.feat_drop = nn.Dropout(dropout)
        self.scale = np.sqrt(hidden_dim) * num_heads  # 左侧为固定缩放因子，右侧则用于对多头注意力求平均

    def forward(self, subgraph, feat_src, feat_dst):
        with subgraph.local_scope():
            feat_src = self.feat_drop(feat_src)
            feat_dst = self.feat_drop(feat_dst)
            src_proj = self.Q_proj(feat_src)
            dst_proj = self.K_proj(feat_dst)
            subgraph.srcdata.update({'src_proj': src_proj})
            subgraph.dstdata.update({'dst_proj': dst_proj})
            subgraph.apply_edges(dgl.function.u_dot_v('src_proj', 'dst_proj', 'score'))  # 相当于求出每个头的注意力并求和
            score = subgraph.edata['score'] / self.scale
            return score

    def predict(self, src_feats, dst_feats):
        src_proj = self.Q_proj(src_feats)
        dst_proj = self.K_proj(dst_feats)
        attention_scores = torch.sum(src_proj * dst_proj, dim=1, keepdim=True)  # (n,1)
        attention_scores = attention_scores / self.scale
        return attention_scores


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)
