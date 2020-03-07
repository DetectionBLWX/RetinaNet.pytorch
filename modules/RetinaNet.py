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


'''build target layer'''
class buildTargetLayer(nn.Module):
	def __init__(self, cfg, **kwargs):
		super(buildTargetLayer, self).__init__()
		self.allowed_border = 0
		self.fg_iou_thresh = cfg.FG_IOU_THRESH
		self.bg_iou_thresh = cfg.BG_IOU_THRESH
		self.bbox_normalize_means = torch.FloatTensor(cfg.BBOX_NORMALIZE_MEANS)
		self.bbox_normalize_stds = torch.FloatTensor(cfg.BBOX_NORMALIZE_STDS)
	'''forward'''
	def forward(self, x):
		# parse x
		anchors, gt_boxes, img_info, num_gt_boxes = x
		batch_size = gt_boxes.size(0)
		self.bbox_normalize_means = self.bbox_normalize_means.type_as(gt_boxes)
		self.bbox_normalize_stds = self.bbox_normalize_stds.type_as(gt_boxes)
		# record number of anchors
		total_anchors_ori = anchors.size(0)
		# filter anchors
		keep_idxs = ((anchors[:, 0] >= -self.allowed_border) &
					 (anchors[:, 1] >= -self.allowed_border) &
					 (anchors[:, 2] < int(img_info[0][1])+self.allowed_border) &
					 (anchors[:, 3] < int(img_info[0][0])+self.allowed_border))
		keep_idxs = torch.nonzero(keep_idxs).view(-1)
		anchors = anchors[keep_idxs, :]
		# prepare classification targets: larger than 0 denotes for objects, 0 denotes for background, -1 means ignore
		cls_targets = gt_boxes.new(batch_size, keep_idxs.size(0)).fill_(-1)
		# prepare regression targets
		reg_targets = gt_boxes.new(batch_size, keep_idxs.size(0), 4).fill_(0)
		# build targets
		for batch_idx in range(batch_size):
			anchors_each = anchors.clone()
			gt_boxes_each = gt_boxes[batch_idx][:num_gt_boxes[batch_idx].int().item()]
			overlaps = BBoxFunctions.calcIoUs(anchors_each, gt_boxes_each[:, :-1])
			max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
			gt_max_overlaps, gt_argmax_overlaps = torch.max(overlaps, 0)
			max_overlaps.index_fill_(0, gt_argmax_overlaps, 2)
			for i in range(gt_argmax_overlaps.size(0)):
				argmax_overlaps[gt_argmax_overlaps[i]] = i
			cls_targets_each = gt_boxes_each[:, -1][argmax_overlaps]
			# --add background and ignore
			cls_targets_each[max_overlaps.lt(self.fg_iou_thresh)] = 0
			cls_targets_each[max_overlaps.lt(self.fg_iou_thresh) & max_overlaps.gt(self.bg_iou_thresh)] = -1
			# --encode boxes
			reg_targets_each = gt_boxes_each[:, :-1][argmax_overlaps]
			reg_targets_each = BBoxFunctions.encodeBboxes(anchors_each, reg_targets_each)
			# --merge
			cls_targets[batch_idx] = cls_targets_each
			reg_targets[batch_idx] = reg_targets_each
		# post-processing
		reg_targets = ((reg_targets - self.bbox_normalize_means.expand_as(reg_targets)) / self.bbox_normalize_stds.expand_as(reg_targets))
		# unmap
		reg_targets = buildTargetLayer.unmap(reg_targets, total_anchors_ori, keep_idxs, batch_size, fill=-1)
		cls_targets = buildTargetLayer.unmap(cls_targets, total_anchors_ori, keep_idxs, batch_size, fill=0)
		# pack return values into outputs and return them
		outputs = [cls_targets, reg_targets]
		return outputs
	'''unmap'''
	@staticmethod
	def unmap(data, count, inds, batch_size, fill=0):
		if data.dim() == 2:
			ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
			ret[:, inds] = data
		else:
			ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
			ret[:, inds, :] = data
		return ret
	'''no backward'''
	def backward(self, *args):
		pass


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
		# define build target layer
		self.build_target_layer = None
		# define regression and classification layer
		self.regression_layer = None
		self.classification_layer = None
		# define the focal loss layer
		self.focal_loss = None
	'''forward'''
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		feature_shapes = []
		# get fpn features
		features = self.fpn_model(x)
		# get regression prediction
		preds_reg = []
		for x in features:
			# --record the shape of each feature map
			feature_shapes.append([x.size(2), x.size(3)])
			out = self.regression_layer(x)
			# --convert (B, C, H, W) to (B, H, W, C)
			out = out.permute(0, 2, 3, 1)
			# --convert (B, H, W, C) to (B, H*W*num_anchors, 4)
			out = out.contiguous().view(batch_size, -1, 4)
			# --append
			preds_reg.append(out)
		preds_reg = torch.cat(preds_reg, dim=1)
		# get classification prediction
		preds_cls = []
		for x in features:
			out = self.classification_layer(x)
			# --convert (B, C, H, W) to (B, H, W, C)
			out = out.permute(0, 2, 3, 1)
			# --convert (B, H, W, C) to (B, H, W, num_anchors, num_classes)
			out = out.contiguous().view(batch_size, out.size(1), out.size(2), self.num_anchors, self.num_classes)
			# --convert (B, H, W, num_anchors, num_classes) to (B, H*W*num_anchors, num_classes)
			out = out.view(batch_size, -1, self.num_classes)
			# --append
			preds_cls.append(out)
		preds_cls = torch.cat(preds_cls, dim=1)
		# get anchors
		anchors = RetinanetBase.generateAnchors(base_sizes=self.anchor_base_sizes, scales=self.anchor_scales, ratios=self.anchor_ratios, feature_shapes=feature_shapes, feature_strides=self.feature_strides)
		anchors = anchors.type_as(x)
		# define losses
		loss_cls = torch.Tensor([0]).type_as(x)
		loss_reg = torch.Tensor([0]).type_as(x)
		# if mode == 'TRAIN', calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			targets = self.build_target_layer((anchors.data, gt_boxes, img_info, num_gt_boxes))
			cls_targets, reg_targets = targets
			# --calculate regression loss
			if self.cfg.REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				loss_reg = betaSmoothL1Loss(bbox_preds=preds_reg[cls_targets>0].view(-1, 4), 
											bbox_targets=reg_targets[cls_targets>0].view(-1, 4), 
											beta=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['beta'], 
											size_average=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['size_average'],
											loss_weight=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['weight'])
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.REG_LOSS_SET['type'])
			# --calculate classification loss
			if self.cfg.CLS_LOSS_SET['type'] == 'focal_loss':
				cls_targets_filtered = cls_targets[cls_targets > -1].view(-1)
				preds_cls_filtered = preds_cls[cls_targets > -1]
				cls_targets_filtered_one_hot = cls_targets_filtered.new(*cls_targets_filtered.size(), self.num_classes).fill_(0)
				cls_targets_filtered_one_hot.scatter_(1, cls_targets_filtered.unsqueeze(-1).long(), 1)
				loss_cls = self.focal_loss(preds_cls_filtered.view(-1, 1), cls_targets_filtered_one_hot.view(-1, 1).long())
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.CLS_LOSS_SET['type'])
		# return the necessary data
		return anchors, nn.Sigmoid()(preds_cls), preds_reg, loss_cls, loss_reg
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
		assert (len(base_sizes) == len(feature_shapes)) and (len(feature_shapes) == len(feature_strides)), 'for <base_sizes> <feature_shapes> and <feature_strides>, expect the same length.'
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
		# define build target layer
		self.build_target_layer = buildTargetLayer(cfg)
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
		# define the focal loss layer
		self.focal_loss = FocalLoss(gamma=cfg.CLS_LOSS_SET['focal_loss']['gamma'], 
									alpha=cfg.CLS_LOSS_SET['focal_loss']['alpha'], 
									size_average=cfg.CLS_LOSS_SET['focal_loss']['size_average'], 
									loss_weight=cfg.CLS_LOSS_SET['focal_loss']['weight'])
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