#! /usr/bin/env python3


import torch
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


def SNR_dist(x, y, dim):
    return torch.var(x-y, dim=dim) / torch.var(x, dim=dim)


class SignalToNoiseRatioContrastiveLoss(BaseMetricLossFunction):

    def __init__(self, pos_margin, neg_margin, regularizer_weight, avg_non_zero_only=True, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.regularizer_weight = regularizer_weight
        self.avg_non_zero_only = avg_non_zero_only
        self.add_to_recordable_attributes(list_of_names=["num_non_zero_pos_pairs", "num_non_zero_neg_pairs", "feature_distance_from_zero_mean_distribution"])
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        a1, p, a2, n = lmu.convert_to_pairs(indices_tuple, labels)
        pos_loss, neg_loss, reg_loss = 0, 0, 0
        if len(a1) > 0:
            pos_loss, self.num_non_zero_pos_pairs = self.mask_margin_and_calculate_loss(embeddings[a1], embeddings[p], labels[a1], self.pos_margin, 1)
        if len(a2) > 0:
            neg_loss, self.num_non_zero_neg_pairs = self.mask_margin_and_calculate_loss(embeddings[a2], embeddings[n], labels[a2], self.neg_margin, -1)
        self.feature_distance_from_zero_mean_distribution = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))
        if self.regularizer_weight > 0:
            reg_loss = self.regularizer_weight * self.feature_distance_from_zero_mean_distribution
        return pos_loss + neg_loss + reg_loss


    def mask_margin_and_calculate_loss(self, anchors, others, labels, margin, before_relu_multiplier):
        d = SNR_dist(anchors, others, dim=1)
        d = torch.nn.functional.relu((d-margin)*before_relu_multiplier)
        num_non_zero_pairs = (d > 0).nonzero().size(0)
        if self.avg_non_zero_only:
            if num_non_zero_pairs > 0:
                d = torch.sum(d) / num_non_zero_pairs
            else:
                d = 0
        else:
            d = torch.mean(d)
        return d, num_non_zero_pairs