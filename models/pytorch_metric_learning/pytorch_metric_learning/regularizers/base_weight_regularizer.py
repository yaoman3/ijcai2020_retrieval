#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f

class BaseWeightRegularizer(torch.nn.Module):
    def __init__(self, normalize_weights=True):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.add_to_recordable_attributes(name="avg_weight_norm")

    def compute_loss(self, weights):
        raise NotImplementedError

    def forward(self, weights):
        """
        weights should have shape (num_classes, embedding_size)
        """
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        self.weight_norms = torch.norm(weights, p=2, dim=1)
        self.avg_weight_norm = torch.mean(self.weight_norms)
        loss = self.compute_loss(weights)
        if loss == 0:
            loss = torch.sum(weights*0)
        return loss

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)