'''
Function:
	define the focal loss
Author:
	Charles
'''
import torch.nn as nn
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