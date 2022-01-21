import torch
import numpy as np
def Focalloss(predictions, labels, weights=None, alpha=0.25, gamma=2):


    """Compute focal loss for predictions.
    Multi-labels Focal loss formula:
    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
            ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
    predictions: A float tensor of shape [batch_size, 
    num_classes] representing the predicted logits for each class
    target_tensor: A float tensor of shape [batch_size,
    num_classes] representing one-hot encoded classification targets
    weights: A float tensor of shape [batch_size]
    alpha: A scalar tensor for focal loss alpha hyper-parameter
    gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
    loss: A (scalar) tensor representing the value of the loss function
    """ 
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.

    zeros = torch.zeros_like(predictions, dtype=predictions.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = torch.where(labels > zeros, labels - predictions, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = torch.where(labels > zeros, zeros, predictions)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(predictions, 1e-8, 1.0)) \
                            - (1 - alpha) * (neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - predictions, 1e-8, 1.0))
    return torch.mean(torch.sum(per_entry_cross_ent, 1))
