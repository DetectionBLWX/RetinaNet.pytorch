'''
Function:
	define the Retinanet
Author:
	Charles
'''
import torch
import torch.nn as nn
from modules.backbones import *
from modules.losses.smoothL1 import *
from modules.losses.focalLoss import *


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
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		# get fpn features
		features = self.fpn_model(x)
		# get regression features
		features_reg = []
		for x in features:
			out = self.regression_layer(x)
			# --convert (B, C, H, W) to (B, H, W, C)
			out = out.permute(0, 2, 3, 1)
			# --convert (B, H, W, C) to (B, H*W*num_anchors, 4)
			out = out.contiguous().view(batch_size, -1, 4)
			# --append
			features_reg.append(out)
		features_reg = torch.cat(features_reg, dim=1)
		# get classification features
		features_cls = []
		for x in features:
			out = self.classification_layer(x)
			# --convert (B, C, H, W) to (B, H, W, C)
			out = out.permute(0, 2, 3, 1)
			# --convert (B, H, W, C) to (B, H, W, num_anchors, num_classes)
			out = out.contiguous().view(batch_size, out.size(1), out.size(2), self.num_anchors, self.num_classes)
			# --convert (B, H, W, num_anchors, num_classes) to (B, H*W*num_anchors, num_classes)
			out = out.view(batch_size, -1, self.num_classes)
			# --append
			features_cls.append(out)
		features_cls = torch.cat(features_cls, dim=1)
		# get anchors



	'''initialize except for backbone network'''
	def initializeAddedModules(self):
		raise NotImplementedError
	'''generate anchors'''
	@staticmethod
	def generateAnchors(self):
		pass
	'''set bn fixed'''
	@staticmethod
	def setBnFixed(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			for p in m.parameters():
				p.requires_grad = False
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
												 nn.Conv2d(in_channels=256, out_channels=self.num_anchors*self.num_classes, kernel_size=3, stride=1, padding=1),
												 nn.Sigmoid())
		# weights initialize
		if mode == 'TRAIN' and cfg.ADDED_MODULES_WEIGHT_INIT_METHOD:
			self.initializeAddedModules()
		# fixed bn
		self.fpn_model.apply(RetinanetBase.setBnFixed)
		self.regression_layer.apply(RetinanetBase.setBnFixed)
		self.classification_layer.apply(RetinanetBase.setBnFixed)
	'''set train mode'''
	def setTrain(self):
		nn.Module.train(self, True)
		self.fpn_model.apply(RetinanetBase.setBnEval)
		self.regression_layer.apply(RetinanetBase.setBnEval)
		self.classification_layer.apply(RetinanetBase.setBnEval)