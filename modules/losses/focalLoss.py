'''
Function:
	define the focal loss
Author:
	Charles
'''
import torch.nn as nn
import torch.nn.functional as F
from libs.sigmoid_focal_loss.sigmoid_focal_loss import sigmoid_focal_loss


'''define the focal loss'''
class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha=0.25, size_average=True, loss_weight=1.0, **kwargs):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.size_average = size_average
		self.loss_weight = loss_weight
	def forward(self, preds, targets):
		loss = sigmoid_focal_loss(preds, targets, self.gamma, self.alpha)
		loss = loss.mean() if self.size_average else loss.sum()
		return self.loss_weight * loss


'''for test'''
def pySigmoidFocalLoss(preds, targets, loss_weight=1.0, gamma=2.0, alpha=0.25, size_average=True):
	preds_sigmoid = preds.sigmoid()
	targets = targets.type_as(preds)
	pt = (1 - preds_sigmoid) * targets + preds_sigmoid * (1 - targets)
	focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)
	loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none') * focal_weight
	loss = loss.mean() if size_average else loss.sum()
	return loss * loss_weight