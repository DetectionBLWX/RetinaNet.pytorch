'''
Function:
	define the Retinanet
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
from modules.utils import *
from modules.losses import *
from modules.backbones import *


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
		# build targets for anchors in each image
		for batch_idx in range(batch_size):
			anchors_single = anchors.clone()
			gt_boxes_single = gt_boxes[batch_idx][:num_gt_boxes[batch_idx].int().item()]
			overlaps = BBoxFunctions.calcIoUs(gt_boxes_single[:, :4].data, anchors_single.data)
			num_gts, num_anchors = overlaps.size(0), overlaps.size(1)
			max_overlaps, argmax_overlaps = overlaps.max(dim=0)
			gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
			# --assign -1 by default
			assign_gts_to_anchors = overlaps.new_full((num_anchors, ), -1, dtype=torch.long)
			# --assign background
			assign_gts_to_anchors[(max_overlaps >= 0) & (max_overlaps < self.bg_iou_thresh)] = 0
			# --assign objects
			pos_idxs = (max_overlaps >= self.fg_iou_thresh)
			assign_gts_to_anchors[pos_idxs] = argmax_overlaps[pos_idxs] + 1
			for i in range(num_gts):
				if gt_max_overlaps[i] >= 0:
					max_iou_idxs = (overlaps[i, :] == gt_max_overlaps[i])
					assign_gts_to_anchors[max_iou_idxs] = i + 1
			# --obtain targets for classification
			pos_idxs = assign_gts_to_anchors > 0
			neg_idxs = assign_gts_to_anchors == 0
			cls_targets[batch_idx][pos_idxs] = gt_boxes_single[:, -1][assign_gts_to_anchors[pos_idxs] - 1]
			cls_targets[batch_idx][neg_idxs] = 0.
			# --obtain targets for regression
			reg_targets[batch_idx][pos_idxs] = BBoxFunctions.encodeBboxes(anchors_single[pos_idxs], gt_boxes_single[:, :4][assign_gts_to_anchors[pos_idxs] - 1])
		# post-processing
		reg_targets = ((reg_targets - self.bbox_normalize_means.expand_as(reg_targets)) / self.bbox_normalize_stds.expand_as(reg_targets))
		# unmap
		reg_targets = buildTargetLayer.unmap(reg_targets, total_anchors_ori, keep_idxs, batch_size, fill=0)
		cls_targets = buildTargetLayer.unmap(cls_targets, total_anchors_ori, keep_idxs, batch_size, fill=-1)
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
		self.num_classes = cfg.NUM_CLASSES - 1
		self.num_anchors = len(cfg.ANCHOR_RATIOS) * len(cfg.ANCHOR_SCALES)
		# define fpn
		self.fpn_model = None
		# define the anchor generators
		self.anchor_generators = None
		# define build target layer
		self.build_target_layer = None
		# define regression and classification layer
		self.reg_layers = None
		self.cls_layers = None
		# define the focal loss layer
		self.focal_loss = None
	'''forward'''
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		feature_shapes = []
		# get fpn features
		features = self.fpn_model(x)
		# do final regression and classification
		preds_reg_list, preds_cls_list = [], []
		for x in features:
			# --record the shape of each feature map
			feature_shapes.append([x.size(2), x.size(3)])
			# --feed into the shared regression layers
			preds_reg_lvl = self.reg_layers(x)
			# --convert (B, *4, H, W) to (B, H, W, num_anchors*4) to (B, H*W*num_anchors, 4)
			preds_reg_lvl = preds_reg_lvl.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
			# --append
			preds_reg_list.append(preds_reg_lvl)
			# --feed into the shared classification layers
			preds_cls_lvl = self.cls_layers(x)
			# --convert (B, num_anchors*num_classes, H, W) to (B, H, W, num_anchors*num_classes) to (B, H*W*num_anchors, num_classes)
			preds_cls_lvl = preds_cls_lvl.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
			# --append
			preds_cls_list.append(preds_cls_lvl)
		# get anchors
		anchors = [generator.generate(feature_shape=shape, feature_stride=stride, device=x.device) for (generator, shape, stride) in zip(self.anchor_generators, feature_shapes, self.feature_strides)]
		num_anchors_levels = [a.size(0) for a in anchors]
		anchors = torch.cat(anchors, 0).type_as(x)
		# define losses
		loss_cls = torch.Tensor([0]).type_as(x)
		loss_reg = torch.Tensor([0]).type_as(x)
		# if mode == 'TRAIN', calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			targets = self.build_target_layer((anchors.data, gt_boxes, img_info, num_gt_boxes))
			cls_targets, reg_targets = targets
			avg_factor = (cls_targets > 0).sum()
			loss_reg_list, loss_cls_list, pointer = [], [], 0
			for lvl_idx, num_anchors_lvl in enumerate(num_anchors_levels):
				preds_reg_lvl = preds_reg_list[lvl_idx]
				preds_cls_lvl = preds_cls_list[lvl_idx]
				cls_targets_lvl = cls_targets[:, pointer: pointer+num_anchors_lvl]
				reg_targets_lvl = reg_targets[:, pointer: pointer+num_anchors_lvl, :]
				pointer = pointer + num_anchors_lvl
				# --calculate regression loss
				if self.cfg.REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
					loss_reg_lvl = betaSmoothL1Loss(bbox_preds=preds_reg_lvl[cls_targets_lvl>0].view(-1, 4),
													bbox_targets=reg_targets_lvl[cls_targets_lvl>0].view(-1, 4), 
													beta=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['beta'], 
													size_average=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['size_average'],
													loss_weight=self.cfg.REG_LOSS_SET['betaSmoothL1Loss']['weight'],
													avg_factor=avg_factor)
				else:
					raise ValueError('Unkown regression loss type <%s>...' % self.cfg.REG_LOSS_SET['type'])
				loss_reg_list.append(loss_reg_lvl)
				# --calculate classification loss
				if self.cfg.CLS_LOSS_SET['type'] == 'focal_loss':
					cls_targets_lvl_filtered = cls_targets_lvl[cls_targets_lvl > -1].view(-1)
					preds_cls_lvl_filtered = preds_cls_lvl[cls_targets_lvl > -1].view(-1, self.num_classes)
					loss_cls_lvl = self.focal_loss(preds=preds_cls_lvl_filtered, 
												   targets=cls_targets_lvl_filtered.long(),
												   avg_factor=avg_factor)
				else:
					raise ValueError('Unkown classification loss type <%s>...' % self.cfg.CLS_LOSS_SET['type'])
				loss_cls_list.append(loss_cls_lvl)
			loss_reg = sum(loss_reg_list)
			loss_cls = sum(loss_cls_list)
		# return the necessary data
		return anchors, torch.cat(preds_cls_list, dim=1).sigmoid(), torch.cat(preds_reg_list, dim=1), loss_cls, loss_reg
	'''initialize except for backbone network'''
	def initializeAddedModules(self, init_method):
		# normal init
		if init_method == 'normal':
			for idx, layer in enumerate(self.reg_layers):
				if isinstance(layer, nn.Conv2d):
					normalInit(layer, std=0.01)
			for idx, layer in enumerate(self.cls_layers):
				if isinstance(layer, nn.Conv2d):
					if idx+1 == len(self.cls_layers):
						normalInit(layer, std=0.01, bias=biasInitWithProb(0.01))
					else:
						normalInit(layer, std=0.01)
		# unsupport
		else:
			raise RuntimeError('Unsupport initializeAddedLayers.init_method <%s>...' % init_method)
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
		# define the anchor generators
		self.anchor_generators = [AnchorGenerator(size_base=size_base, scales=cfg.ANCHOR_SCALES, ratios=cfg.ANCHOR_RATIOS) for size_base in cfg.ANCHOR_BASE_SIZES]
		# define build target layer
		self.build_target_layer = buildTargetLayer(cfg)
		# define regression and classification layer
		self.reg_layers = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=self.num_anchors*4, kernel_size=3, stride=1, padding=1))
		self.cls_layers = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
										nn.ReLU(inplace=True),
										nn.Conv2d(in_channels=256, out_channels=self.num_anchors*self.num_classes, kernel_size=3, stride=1, padding=1))
		# define the focal loss layer
		self.focal_loss = FocalLoss(gamma=cfg.CLS_LOSS_SET['focal_loss']['gamma'], 
									alpha=cfg.CLS_LOSS_SET['focal_loss']['alpha'], 
									size_average=cfg.CLS_LOSS_SET['focal_loss']['size_average'], 
									loss_weight=cfg.CLS_LOSS_SET['focal_loss']['weight'])
		# weights initialize
		if mode == 'TRAIN' and cfg.ADDED_MODULES_WEIGHT_INIT_METHOD:
			self.initializeAddedModules(cfg.ADDED_MODULES_WEIGHT_INIT_METHOD['retina_head'])
			self.fpn_model.initializeAddedLayers(cfg.ADDED_MODULES_WEIGHT_INIT_METHOD['fpn'])
		# fix some first layers following original implementation
		if cfg.FIXED_FRONT_BLOCKS:
			for p in self.fpn_model.base_layer0.parameters():
				p.requires_grad = False
			for p in self.fpn_model.base_layer1.parameters():
				p.requires_grad = False
	'''set train mode'''
	def setTrain(self):
		nn.Module.train(self, True)
		if self.cfg.FIXED_FRONT_BLOCKS:
			for p in self.fpn_model.base_layer0.parameters():
				p.requires_grad = False
			for p in self.fpn_model.base_layer1.parameters():
				p.requires_grad = False
		self.fpn_model.apply(RetinanetBase.setBnEval)
		self.reg_layers.apply(RetinanetBase.setBnEval)
		self.cls_layers.apply(RetinanetBase.setBnEval)