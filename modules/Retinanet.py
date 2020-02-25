'''
Function:
	define the Retinanet
Author:
	Charles
'''
import torch
import torch.nn as nn
from modules.backbones import *


'''base model for Retinanet'''
class RetinanetBase(nn.Module):
	def __init__(self, mode, cfg, **kwargs):
		super(RetinanetBase, self).__init__()
		self.mode = mode
		self.cfg = cfg
		self.num_classes = cfg.NUM_CLASSES
		self.num_anchors = len(cfg.ANCHOR_RATIOS) * len(cfg.ANCHOR_SCALES)
		# define fpn
		self.fpn_model = None
		# define regression and classification layer
		self.regression_layer = None
		self.classification_layer = None
	'''forward'''
	def forward(self, x):
		pass
	'''set bn eval'''
	@staticmethod
	def setBnEval(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.eval()


'''Retinanet using resnet-FPN backbones'''
class RetinanetFPNResNets(RetinanetBase):
	def __init__(self, mode, cfg, logger_handle, **kwargs):
		RetinanetBase.__init__(self, mode, cfg)
		# define fpn
		self.fpn_model = FPNResNets(mode, cfg, logger_handle)
		# define regression and classification layer
		self.regression_layer = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
											  nn.ReLU(inplace=True),
											  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
											  nn.ReLU(inplace=True),
											  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
											  nn.ReLU(inplace=True),
											  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
											  nn.ReLU(inplace=True),
											  nn.Conv2d(in_channels=256, out_channels=self.num_anchors*4, kernel_size=3, stride=1, padding=1))
		self.classification_layer = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
												 nn.ReLU(inplace=True),
												 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
												 nn.ReLU(inplace=True),
												 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
												 nn.ReLU(inplace=True),
												 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
												 nn.ReLU(inplace=True),
												 nn.Conv2d(in_channels=256, out_channels=self.num_anchors*self.num_classes, kernel_size=3, stride=1, padding=1))