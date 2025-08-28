#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 0:45
# @Author  : jc Han
# @help    :
import torch
from config import config
import numpy as np
def delete_batch_idx(items: dict):
    for k, v in items.items():
        if isinstance(v, torch.Tensor):
            assert v.shape[0] == 1
            items[k] = v[0]

    return items

def to_cuda(items: dict, add_batch = False, precision = torch.float32):
    items_cuda = dict()
    for key, data in items.items():
        if isinstance(data, torch.Tensor):
            items_cuda[key] = data.to(config.device)
        elif isinstance(data, np.ndarray):
            items_cuda[key] = torch.from_numpy(data).to(config.device)
        elif isinstance(data, dict):  # usually some float tensors
            for key2, data2 in data.items():
                if isinstance(data2, np.ndarray):
                    data[key2] = torch.from_numpy(data2).to(config.device)
                elif isinstance(data2, torch.Tensor):
                    data[key2] = data2.to(config.device)
                else:
                    raise TypeError('Do not support other data types.')
                if data[key2].dtype == torch.float32 or data[key2].dtype == torch.float64:
                    data[key2] = data[key2].to(precision)
            items_cuda[key] = data
        elif isinstance(data, str):
            items_cuda[key] = data
        else:
            items_cuda[key] = torch.tensor(data, dtype=torch.float32)
        if isinstance(items_cuda[key], torch.Tensor) and\
                (items_cuda[key].dtype == torch.float32 or items_cuda[key].dtype == torch.float64):
            items_cuda[key] = items_cuda[key].to(precision)
        if add_batch:
            if isinstance(items_cuda[key], torch.Tensor):
                items_cuda[key] = items_cuda[key].unsqueeze(0)
            elif isinstance(items_cuda[key], dict):
                for k in items_cuda[key].keys():
                    items_cuda[key][k] = items_cuda[key][k].unsqueeze(0)
            else:
                items_cuda[key] = [items_cuda[key]]
    return items_cuda

