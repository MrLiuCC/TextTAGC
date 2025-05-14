import torch


def merge_features(feature1, feature2, merge_type):
    # 将节点的两种特征向量合并为单个特征向量
    if merge_type == "concat":
        targets = torch.cat((feature1, feature2), dim=1)
    elif merge_type == "add":
        assert feature1.shape == feature2.shape, (
            "feature1与feature2的形状不相同：{} 和 {}".format(feature1.shape, feature2.shape))
        targets = feature1 + feature2
    elif merge_type == "none":
        # 不合并
        targets = feature1
    else:
        # 不支持的merge_type类型
        raise ValueError(f"Invalid merge type: {merge_type}. "
                         f"Only 'add', 'concat' and 'none' are supported.")
    return targets


def merge_time_encoding(features, timestamps, time_encoder, time_combination: str):
    # 将特征向量与其时序编码向量结合，输出新的特征向量
    time_combination = time_combination.lower()
    assert timestamps.size()[0] == features.size()[0], \
        (f"The first dimensions of features[{str(features.size()[0])}] and timestamps[{str(timestamps.size()[0])}] "
         f"don't match")
    time_encoding = time_encoder(timestamps.float())
    # 将特征向量与时序编码结合
    combine_feats = merge_features(features, time_encoding, time_combination)
    return combine_feats
