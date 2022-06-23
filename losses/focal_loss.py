"""
Focal Loss implementation.
reference paper: https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    reference paper: https://arxiv.org/abs/1708.02002
    """

    length = len(cls_num_list)
    per_cls_weights = np.zeros(length)
    cls_num_list = np.array(cls_num_list)
    # normalization factor
    f = 1/(np.sum(1-beta**cls_num_list)/(1-beta))*length
    for i in range(length):
        per_cls_weights[i] = f*(1-beta**cls_num_list[i])/(1-beta)
    per_cls_weights = torch.from_numpy(per_cls_weights)
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        m = nn.Sigmoid()
        gamma = self.gamma
        num_cls = input.shape[1]
        class_range = torch.arange(0, num_cls).unsqueeze(0)
        if torch.cuda.is_available():
            class_range = class_range.cuda()
        t = target.unsqueeze(1)
        p = m(input)
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        loss_tmp = -(t == class_range).float() * term1 - ((t != class_range) * (t >= 0)).float() * term2
                    
        loss = torch.mean(self.weight*loss_tmp)

        return loss
