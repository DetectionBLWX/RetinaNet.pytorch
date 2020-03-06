'''
Function:
	some util functions used for many module files
Author:
	Charles
'''
import os
import torch
import logging


'''check the existence of dirpath'''
def checkDir(dirpath):
	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
		return False
	return True


'''log function.'''
class Logger():
	def __init__(self, logfilepath, **kwargs):
		logging.basicConfig(level=logging.INFO,
							format='%(asctime)s %(levelname)-8s %(message)s',
							datefmt='%Y-%m-%d %H:%M:%S',
							handlers=[logging.FileHandler(logfilepath),
									  logging.StreamHandler()])
	@staticmethod
	def log(level, message):
		logging.log(level, message)
	@staticmethod
	def debug(message):
		Logger.log(logging.DEBUG, message)
	@staticmethod
	def info(message):
		Logger.log(logging.INFO, message)
	@staticmethod
	def warning(message):
		Logger.log(logging.WARNING, message)
	@staticmethod
	def error(message):
		Logger.log(logging.ERROR, message)


'''load class labels.'''
def loadclsnames(clsnamespath):
	names = []
	for line in open(clsnamespath):
		if line.strip('\n'):
			names.append(line.strip('\n'))
	return names


'''adjust learning rate'''
def adjustLearningRate(optimizer, target_lr, logger_handle):
	logger_handle.info('Adjust learning rate to %s...' % str(target_lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = target_lr
	return True


'''some functions for bboxes, the format of all the input bboxes are (x1, y1, x2, y2)'''
class BBoxFunctions(object):
	def __init__(self):
		self.info = 'bbox functions'
	def __repr__(self):
		return self.info
	'''clip boxes, boxes size: B x N x 4, img_info: B x 3(height, width, scale_factor)'''
	@staticmethod
	def clipBoxes(boxes, img_info):
		for i in range(boxes.size(0)):
			boxes[i, :, 0::4].clamp_(0, img_info[i, 1]-1)
			boxes[i, :, 1::4].clamp_(0, img_info[i, 0]-1)
			boxes[i, :, 2::4].clamp_(0, img_info[i, 1]-1)
			boxes[i, :, 3::4].clamp_(0, img_info[i, 0]-1)
		return boxes
	'''calculate iou, boxes1(anchors): N x 4, boxes2(gts): K x 4'''
	@staticmethod
	def calcIoUs(boxes1, boxes2, eps=1e-10):
		num_boxes1 = boxes1.size(0)
		num_boxes2 = boxes2.size(0)
		# calc intersect
		max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(num_boxes1, num_boxes2, 2), boxes2[:, 2:].unsqueeze(0).expand(num_boxes1, num_boxes2, 2))
		min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(num_boxes1, num_boxes2, 2), boxes2[:, :2].unsqueeze(0).expand(num_boxes1, num_boxes2, 2))
		inter = torch.clamp((max_xy - min_xy), min=0)
		inter = inter[..., 0] * inter[..., 1]
		# calc union
		area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])).unsqueeze(1).expand_as(inter)
		area2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])).unsqueeze(0).expand_as(inter)
		union = area1 + area2 - inter
		# calc ious, the size of overlaps is (N, K, 4)
		overlaps = inter / (union + eps)
		# return overlaps
		return overlaps
	'''encode bboxes'''
	@staticmethod
	def encodeBboxes(boxes_pred, boxes_gt):
		if boxes_pred.dim() == 2:
			# convert (x1, y1, x2, y2) to (cx, cy, w, h) 
			widths_pred = boxes_pred[..., 2] - boxes_pred[..., 0] + 1.0
			heights_pred = boxes_pred[..., 3] - boxes_pred[..., 1] + 1.0
			centerxs_pred = boxes_pred[..., 0] + 0.5 * widths_pred
			centerys_pred = boxes_pred[..., 1] + 0.5 * heights_pred
			widths_gt = boxes_gt[..., 2] - boxes_gt[..., 0] + 1.0
			heights_gt = boxes_gt[..., 3] - boxes_gt[..., 1] + 1.0
			centerxs_gt = boxes_gt[..., 0] + 0.5 * widths_gt
			centerys_gt = boxes_gt[..., 1] + 0.5 * heights_gt
			# calculate targets
			dxs_target = (centerxs_gt - centerxs_pred.view(1, -1).expand_as(centerxs_gt)) / widths_pred
			dys_target = (centerys_gt - centerys_pred.view(1, -1).expand_as(centerys_gt)) / heights_pred
			dws_target = torch.log(widths_gt / widths_pred.view(1, -1).expand_as(widths_gt))
			dhs_target = torch.log(heights_gt / heights_pred.view(1, -1).expand_as(heights_gt))
		elif boxes_pred.dim() == 3:
			# convert (x1, y1, x2, y2) to (cx, cy, w, h) 
			widths_pred = boxes_pred[..., 2] - boxes_pred[..., 0] + 1.0
			heights_pred = boxes_pred[..., 3] - boxes_pred[..., 1] + 1.0
			centerxs_pred = boxes_pred[..., 0] + 0.5 * widths_pred
			centerys_pred = boxes_pred[..., 1] + 0.5 * heights_pred
			widths_gt = boxes_gt[..., 2] - boxes_gt[..., 0] + 1.0
			heights_gt = boxes_gt[..., 3] - boxes_gt[..., 1] + 1.0
			centerxs_gt = boxes_gt[..., 0] + 0.5 * widths_gt
			centerys_gt = boxes_gt[..., 1] + 0.5 * heights_gt
			# calculate targets
			dxs_target = (centerxs_gt - centerxs_pred) / widths_pred
			dys_target = (centerys_gt - centerys_pred) / heights_pred
			dws_target = torch.log(widths_gt / widths_pred)
			dhs_target = torch.log(heights_gt / heights_pred)
		else:
			raise ValueError('boxes_pred dimension error in BBoxFunctions.encodeBboxes')
		return torch.stack((dxs_target, dys_target, dws_target, dhs_target), 2)
	'''decode bboxes'''
	@staticmethod
	def decodeBboxes(boxes, deltas):
		widths = boxes[..., 2] - boxes[..., 0] + 1.0
		heights = boxes[..., 3] - boxes[..., 1] + 1.0
		cxs = boxes[..., 0] + 0.5 * widths
		cys = boxes[..., 1] + 0.5 * heights
		dxs = deltas[..., 0::4]
		dys = deltas[..., 1::4]
		dws = deltas[..., 2::4]
		dhs = deltas[..., 3::4]
		cxs_pred = dxs * widths.unsqueeze(2) + cxs.unsqueeze(2)
		cys_pred = dys * heights.unsqueeze(2) + cys.unsqueeze(2)
		ws_pred = torch.exp(dws) * widths.unsqueeze(2)
		hs_pred = torch.exp(dhs) * heights.unsqueeze(2)
		boxes_pred = deltas.clone()
		boxes_pred[..., 0::4] = cxs_pred - ws_pred * 0.5
		boxes_pred[..., 1::4] = cys_pred - hs_pred * 0.5
		boxes_pred[..., 2::4] = cxs_pred + ws_pred * 0.5
		boxes_pred[..., 3::4] = cys_pred + hs_pred * 0.5
		# [x1, y1, x2, y2]
		return boxes_pred


'''save checkpoints'''
def saveCheckpoints(state_dict, savepath, logger_handle):
	logger_handle.info('Saving state_dict in %s...' % savepath)
	torch.save(state_dict, savepath)
	return True


'''load checkpoints'''
def loadCheckpoints(checkpointspath, logger_handle):
	logger_handle.info('Loading checkpoints from %s...' % checkpointspath)
	checkpoints = torch.load(checkpointspath)
	return checkpoints