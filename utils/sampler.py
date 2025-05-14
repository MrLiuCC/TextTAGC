import dgl
import torch


def select_negative_sampler(sampler_type, k):
    sampler_type = sampler_type.lower()
    if sampler_type == "uniform":
        return dgl.dataloading.negative_sampler.Uniform(k)
    # Uniform()此处加上 exclude_positive_edges=True 可以排除正样本，避免采到假负样本
    elif sampler_type == "time":
        return TemporalNegativeSampler(k)
    else:
        raise ValueError(f"Invalid Negative sampler type: {sampler_type}. "
                         f"Only 'uniform' and 'time' are supported.")


class TemporalNegativeSampler:
    def __init__(self, k, timing_attr="year"):
        self.k = k
        self.timing_attr = timing_attr

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        # 获取源节点的时序属性，并计算其中的最大值
        max_timing = torch.max(g.ndata[self.timing_attr][src])
        src = src.repeat_interleave(self.k)

        device = g.device
        # 负采样的候选节点仅包含时序属性 <= max_timing的节点
        candidate_nodes = torch.nonzero(g.ndata[self.timing_attr] <= max_timing, as_tuple=False).squeeze().to(device)
        # 随机选取
        dst_index = torch.randint(0, candidate_nodes.size(0), (src.size(0),), device=device)
        dst = candidate_nodes[dst_index]

        return src, dst
