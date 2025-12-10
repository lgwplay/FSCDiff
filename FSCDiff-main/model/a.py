import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, Any, List, Tuple

def normalize_to_01(tensor):
    """将tensor归一化到[0,1]范围"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)

def simple_train_val_forward(model: nn.Module, gt=None, image=None, dp=None, **kwargs):
    """原始的训练/验证前向传播函数"""
    if model.training:
        assert gt is not None and image is not None and dp is not None
        return model(gt, image, dp, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, dp, **kwargs)
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)
            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.6).float()
                p = p_postion * p
                return p
            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "dp": dp,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }



# 示例用法:
# model = YourModel()
# results = analyze_model_performance(model, input_shape=(1, 3, 512, 512))