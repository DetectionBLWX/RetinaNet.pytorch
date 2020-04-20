'''
Function:
	some util functions used for many module files
Author:
	Charles
'''
import os
import torch
import logging
import numpy as np
from torch.nn.utils import clip_grad


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
	'''clip bboxes, bboxes size: B x N x 4, img_info: B x 3(height, width, scale_factor)'''
	@staticmethod
	def clipBoxes(bboxes, img_info):
		assert bboxes.size(0) == img_info.size(0)
		for i in range(bboxes.size(0)):
			bboxes[i, :, 0::4].clamp_(0, img_info[i, 1]-1)
			bboxes[i, :, 1::4].clamp_(0, img_info[i, 0]-1)
			bboxes[i, :, 2::4].clamp_(0, img_info[i, 1]-1)
			bboxes[i, :, 3::4].clamp_(0, img_info[i, 0]-1)
		return bboxes
	'''calculate ious, bboxes1: N x 4, bboxes2: K x 4'''
	@staticmethod
	def calcIoUs(bboxes1, bboxes2, is_aligned=False):
		num_bboxes1 = bboxes1.size(0)
		num_bboxes2 = bboxes2.size(0)
		assert num_bboxes1 * num_bboxes2 != 0
		if is_aligned: assert num_bboxes1 == num_bboxes2
		if is_aligned:
			lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
			rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
			wh = (rb - lt + 1).clamp(min=0)
			overlap = wh[:, 0] * wh[:, 1]
			area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
			area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
			ious = overlap / (area1 + area2 - overlap)
		else:
			lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
			rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
			wh = (rb - lt + 1).clamp(min=0)
			overlap = wh[:, :, 0] * wh[:, :, 1]
			area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
			area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
			ious = overlap / (area1[:, None] + area2 - overlap)
		return ious
	'''encode bboxes'''
	@staticmethod
	def encodeBboxes(preds, gts):
		assert preds.size() == gts.size()
		preds, gts = preds.float(), gts.float()
		# preds
		px = (preds[..., 0] + preds[..., 2]) * 0.5
		py = (preds[..., 1] + preds[..., 3]) * 0.5
		pw = preds[..., 2] - preds[..., 0] + 1.0
		ph = preds[..., 3] - preds[..., 1] + 1.0
		# gts
		gx = (gts[..., 0] + gts[..., 2]) * 0.5
		gy = (gts[..., 1] + gts[..., 3]) * 0.5
		gw = gts[..., 2] - gts[..., 0] + 1.0
		gh = gts[..., 3] - gts[..., 1] + 1.0
		# deltas
		dx = (gx - px) / pw
		dy = (gy - py) / ph
		dw = torch.log(gw / pw)
		dh = torch.log(gh / ph)
		deltas = torch.stack([dx, dy, dw, dh], dim=-1)
		# return values
		return deltas
	'''decode bboxes'''
	@staticmethod
	def decodeBboxes(preds, deltas, wh_ratio_clip=16/1000):
		# deltas
		dx = deltas[..., 0::4]
		dy = deltas[..., 1::4]
		dw = deltas[..., 2::4]
		dh = deltas[..., 3::4]
		max_ratio = np.abs(np.log(wh_ratio_clip))
		dw = dw.clamp(min=-max_ratio, max=max_ratio)
		dh = dh.clamp(min=-max_ratio, max=max_ratio)
		# preds
		px = ((preds[..., 0] + preds[..., 2]) * 0.5).unsqueeze(-1).expand_as(dx)
		py = ((preds[..., 1] + preds[..., 3]) * 0.5).unsqueeze(-1).expand_as(dy)
		pw = (preds[..., 2] - preds[..., 0] + 1.0).unsqueeze(-1).expand_as(dw)
		ph = (preds[..., 3] - preds[..., 1] + 1.0).unsqueeze(-1).expand_as(dh)
		# gts
		gw = pw * dw.exp()
		gh = ph * dh.exp()
		gx = torch.addcmul(px, 1, pw, dx)
		gy = torch.addcmul(py, 1, ph, dy)
		x1 = gx - gw * 0.5 + 0.5
		y1 = gy - gh * 0.5 + 0.5
		x2 = gx + gw * 0.5 - 0.5
		y2 = gy + gh * 0.5 - 0.5
		bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
		# return values
		return bboxes


'''save checkpoints'''
def saveCheckpoints(state_dict, savepath, logger_handle):
	logger_handle.info('Saving state_dict in %s...' % savepath)
	torch.save(state_dict, savepath)
	return True


'''load checkpoints'''
def loadCheckpoints(checkpointspath, logger_handle):
	logger_handle.info('Loading checkpoints from %s...' % checkpointspath)
	checkpoints = torch.load(checkpointspath, map_location=torch.device('cpu'))
	return checkpoints


'''clip gradient'''
def clipGradients(params, max_norm=35, norm_type=2):
	params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
	if len(params) > 0:
		clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)