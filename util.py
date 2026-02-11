
import os
import random
import numpy as np
import torch
import torch.nn.functional as F


EPS=1e-6

def calculate_iou(pred, mask):
    pred_flat = pred.view(pred.size(0), -1)
    mask_flat = mask.view(mask.size(0), -1)

    # Compute intersection and union
    intersection = (pred_flat * mask_flat).sum(dim=1)  # Element-wise multiplication and sum
    union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1) - intersection  # Sum of elements in both masks minus intersection

    # Compute IoU for each instance in the batch
    iou = intersection / union

    return iou.sum()

def calculate_f1(pred, mask):
    # Flatten the tensors
    pred_flat = (pred.view(pred.size(0), -1) > 0.5).float()
    mask_flat = mask.view(mask.size(0), -1)

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = (pred_flat * mask_flat).sum(dim=1)  # Element-wise multiplication for TP
    FP = ((pred_flat == 1) & (mask_flat == 0)).sum(dim=1)  # Pred is 1, Mask is 0
    FN = ((pred_flat == 0) & (mask_flat == 1)).sum(dim=1)  # Pred is 0, Mask is 1

    # Compute Precision and Recall
    precision = TP / (TP + FP + 1e-8)  # Add small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-8)  # Add small epsilon to avoid division by zero

    # Compute F1 Score (harmonic mean of Precision and Recall)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  # Small epsilon to avoid division by zero

    return f1_score.sum(), precision.sum(), recall.sum()  # Mean F1 score across the batch

def calculate_objective(pred, target):
            # Compute Dice Loss
            intersection = (pred.flatten() * target.flatten()).sum()
            dice_loss = 1 - (2. * intersection + EPS) / (pred.flatten().sum() + target.flatten().sum() + EPS)

            _focal_loss = focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean')

            # Combine both losses
            loss = 10 * _focal_loss + dice_loss
            return loss

def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Computes the focal loss between pred and target.
    
    Args:
        pred (Tensor): Predicted probabilities with shape (B, H, W) or (B, 1, H, W).
        target (Tensor): Ground truth binary labels with same shape as pred.
        alpha (float): Balancing factor.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method ('mean' or 'sum').
    
    Returns:
        Tensor: Computed focal loss.
    """
    # Compute binary cross entropy loss without reduction
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Compute p_t (the probability of the true class)
    pt = torch.exp(-bce_loss)
    
    # Compute focal loss
    loss = alpha * (1 - pt) ** gamma * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def log_info(args):
    print('\n:=========== Few-Shot Seg. ===========')
    for arg_key in args.__dict__:
        print('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
    print(':==================================================\n')

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
    return {'trainable': trainable_params, 'all': all_param, 'trainable%': 100 * trainable_params / all_param}