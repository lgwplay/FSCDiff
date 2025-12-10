from torch import nn
import torch.nn.functional as F



def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def simple_train_val_forward(model: nn.Module, gt=None, image=None,dp=None,**kwargs):
    if model.training:
        assert gt is not None and image is not None and dp is not None
        return model(gt,image,dp, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        
        
        results = analyze_model_performance(model, input_shape=(1, 3, 512, 512))
        pred = model.sample(image,dp,**kwargs)
        
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
                # total_steps = len(preds)  # 总的预测步数
                # print(total_steps)
                # weight = weight = (i / total_steps)
                # p = p_postion * p * weight  # 应用权重
                # print(p)
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "dp":dp,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image 





    
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def calculate_binary_change_rate(history_cpu):

    binary_preds = (history_cpu > 0.5).astype(np.float32) 
    changes = np.diff(binary_preds, axis=1) 
    change_rates = np.mean(np.abs(changes), axis=(2, 3)) 
    change_rates = np.insert(change_rates, 0, change_rates[:, 0], axis=1) 
    return change_rates

def modification_train_val_forward(model: nn.Module, gt=None, image=None, dp=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None and dp is not None
        return model(gt, image, dp, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, dp, **kwargs).detach().cpu()
        if time_ensemble:

            preds = torch.concat(model.history, dim=1).detach().cpu().numpy()  

  
            change_rates = calculate_binary_change_rate(preds)

            num_steps_to_select = 5
            if preds.shape[1] < num_steps_to_select:
                raise ValueError("Number of steps is less than the required number of steps to select.")
            

            sorted_indices = np.argsort(change_rates, axis=1)[:, :num_steps_to_select]
            

            selected_preds = np.take_along_axis(preds, sorted_indices[:, :, None, None], axis=1)
            

            ensemble_pred = np.mean(selected_preds, axis=1, keepdims=True)

            pred = torch.from_numpy(ensemble_pred).to(pred.device)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "dp": dp,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }    

    
    


    

def modification_train_val_forward_e(model: nn.Module, gt=None, image=None,dp=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image,dp,seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image,dp,**kwargs).detach().cpu()
        if time_ensemble:
            """ Here is extend function 4, with batch extend."""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            for i in range(2):
                model.sample(image,dp,**kwargs)
                preds = torch.cat([preds, torch.concat(model.history, dim=1).detach().cpu()], dim=1)
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

    
    
    
