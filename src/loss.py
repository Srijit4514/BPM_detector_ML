import torch

def neg_pearson_loss(pred, gt):
    """
    Negative Pearson correlation loss:
    Encourages trend similarity between predicted rPPG curve and ground truth PPG.

    Args:
        pred: Tensor (B, T) predicted rPPG
        gt:   Tensor (B, T) ground truth PPG

    Returns:
        loss: scalar Tensor
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    gt   = gt - gt.mean(dim=1, keepdim=True)

    numerator = torch.sum(pred * gt, dim=1)
    denominator = torch.sqrt(torch.sum(pred**2, dim=1) * torch.sum(gt**2, dim=1) + 1e-8)

    corr = numerator / denominator
    loss = 1 - corr.mean()
    return loss
