import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerGraphConv(nn.Module):
    """Multi layer graph convolution network"""
    def __init__(self, in_feats, hidden_dims=[], num_layers=2):
        super().__init__()
        self.in_feats = in_feats
        self.layers = nn.ModuleList()

        assert len(hidden_dims) == num_layers, \
            f"The size of hidden_dims(={str(len(hidden_dims))}) not equal to num_layers(={str(num_layers)})"

        self.hidden_dims = hidden_dims

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.leaky_relu(hidden_x)
        return hidden_x


class SAGE(MultiLayerGraphConv):
    # multi layer SAGE
    def __init__(self, in_feats, hidden_dims=[], num_layers=2, aggregator_type="mean", dropout=0.1):
        super().__init__(in_feats, hidden_dims, num_layers)
        self.aggregator_type = aggregator_type
        self.dropout = dropout

        # 输入层
        self.layers.append(dglnn.SAGEConv(self.in_feats, self.hidden_dims[0], aggregator_type,
                                          feat_drop=dropout))

        # 隐藏层 + 输出层
        for i in range(1, num_layers):
            self.layers.append(dglnn.SAGEConv(hidden_dims[i - 1], self.hidden_dims[i], aggregator_type,
                                              feat_drop=dropout))


class GAT(MultiLayerGraphConv):
    # multi layer GAT
    def __init__(self, in_feats, hidden_dims=[], num_layers=2, num_heads=1, dropout=0.1):
        super().__init__(in_feats, hidden_dims, num_layers)
        self.num_heads = num_heads
        self.dropout = dropout

        # 输入层
        self.layers.append(dglnn.GATConv(in_feats, self.hidden_dims[0],
                                         num_heads=num_heads, feat_drop=dropout, allow_zero_in_degree=True))

        # 隐藏层 + 输出层
        for i in range(1, num_layers):
            self.layers.append(dglnn.GATConv(hidden_dims[i - 1], self.hidden_dims[i],
                                             num_heads=num_heads, feat_drop=dropout, allow_zero_in_degree=True))

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x).mean(dim=1)
            # 多头注意力输出的形状为(num_nodes, num_heads, out_feats)，根据维度1 取mean后输出维度为（num_nodes, out_feats)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.leaky_relu(hidden_x)

        return hidden_x

