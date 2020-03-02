'''
Function:
	define the Retinanet
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
from modules.backbones import *
from modules.utils.utils import *
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
		self.anchor_base_sizes = cfg.ANCHOR_BASE_SIZES
		self.anchor_ratios = cfg.ANCHOR_RATIOS
		self.anchor_scales = cfg.ANCHOR_SCALES
		# define fpn
		self.fpn_model = None
		# define regression and classification layer
		self.regression_layer = None
		self.classification_layer = None
	'''forward'''
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		feature_shapes = []
		# get fpn features
		features = self.fpn_model(x)
		# get regression features
		features_reg = []
		for x in features:
			# --record the shape of each feature map
			feature_shapes.append([x.size(2), x.size(3)])
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
		anchors = RetinanetBase.generateAnchors(base_sizes=self.anchor_base_sizes, scales=self.anchor_scales, ratios=self.anchor_ratios, feature_shapes=feature_shapes, feature_strides=self.feature_strides)
		# define losses
		loss_cls = torch.Tensor([0]).type_as(x)
		loss_reg = torch.Tensor([0]).type_as(x)
		# if mode == 'TRAIN', calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			pass
		# return the necessary data
		return nn.Sigmoid()(features_cls), features_reg, loss_cls, loss_reg
	'''initialize except for backbone network'''
	def initializeAddedModules(self, init_method):
		raise NotImplementedError
	'''
	Function:
		generate anchors
	Input:
		--base_sizes(list): the base anchor size for each pyramid level.
		--scales(list): scales for each pyramid level.
		--ratios(list): ratios for anchor boxes in each pyramid level.
		--feature_shapes(list): the size of feature maps in each pyramid level.
		--feature_strides(list): the strides in each pyramid level.
	Return:
		--anchors(np.array): [nA, 4], the format is (x1, y1, x2, y2).
	'''
	@staticmethod
	def generateAnchors(base_sizes=[32, 64, 128, 256, 512], scales=[1, 2**(1.0/3.0), 2**(2.0/3.0)], ratios=[0.5, 1, 2], feature_shapes=list(), feature_strides=list()):
		assert len(base_sizes) = len(feature_shapes) and len(feature_shapes) == len(feature_strides), 'for <base_sizes> <feature_shapes> and <feature_strides>, expect the same length.'
		anchors = []
		for i in range(len(base_sizes)):
			for scale in scales:
				scales_pyramid, ratios_pyramid = np.meshgrid(np.array(scale*base_sizes[i]), np.array(ratios))
				scales_pyramid, ratios_pyramid = scales_pyramid.flatten(), ratios_pyramid.flatten()
				heights = scales_pyramid / np.sqrt(ratios_pyramid)
				widths = scales_pyramid * np.sqrt(ratios_pyramid)
				shifts_x = np.arange(0, feature_shapes[i][1], 1) * feature_strides[i] + 0.5 * feature_strides[i]
				shifts_y = np.arange(0, feature_shapes[i][0], 1) * feature_strides[i] + 0.5 * feature_strides[i]
				shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
				widths, cxs = np.meshgrid(widths, shifts_x)
				heights, cys = np.meshgrid(heights, shifts_y)
				boxes_cxcy = np.stack([cxs, cys], axis=2).reshape([-1, 2])
				boxes_whs = np.stack([widths, heights], axis=2).reshape([-1, 2])
				anchors_pyramid = np.concatenate([boxes_cxcy-0.5*boxes_whs, boxes_cxcy+0.5*boxes_whs], axis=1)
				anchors.append(anchors_pyramid)
		anchors = np.concatenate(anchors, axis=0)
		return torch.from_numpy(anchors).float()
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
	feature_strides = [8, 16, 32, 64, 128]
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
			self.initializeAddedModules(cfg.ADDED_MODULES_WEIGHT_INIT_METHOD)
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