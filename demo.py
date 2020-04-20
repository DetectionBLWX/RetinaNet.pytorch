'''
Function:
	detect objects in one image
Author:
	Charles
'''
import os
import torch
import warnings
import argparse
import numpy as np
from modules.utils import *
from libs.nms.nms_wrapper import nms
from PIL import Image, ImageDraw, ImageFont
from modules.RetinaNet import RetinanetFPNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings("ignore")


'''parse arguments in command line'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Faster R-CNN')
	parser.add_argument('--imagepath', dest='imagepath', help='image you want to detect.', default='', type=str, required=True)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for demo.', default='', type=str, required=True)
	parser.add_argument('--datasetname', dest='datasetname', help='dataset used to train.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str, required=True)
	parser.add_argument('--nmsthresh', dest='nmsthresh', help='thresh used in nms.', default=0.5, type=float)
	parser.add_argument('--confthresh', dest='confthresh', help='thresh used in showing bounding box.', default=0.5, type=float)
	args = parser.parse_args()
	return args


'''detect objects in one image'''
def demo():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TEST_BACKUPDIR)
	logger_handle = Logger(cfg.TEST_LOGFILE)
	use_cuda = torch.cuda.is_available()
	clsnames = loadclsnames(cfg.CLSNAMESPATH)
	# prepare model
	if args.backbonename.find('resnet') != -1:
		model = RetinanetFPNResNets(mode='TEST', cfg=cfg, logger_handle=logger_handle)
	else:
		raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
	if use_cuda:
		model = model.cuda()
	# load checkpoints
	checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
	model.load_state_dict(checkpoints['model'])
	model.eval()
	# do detect
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	img = Image.open(args.imagepath)
	if args.datasetname == 'coco':
		input_img, scale_factor, target_size = COCODataset.preprocessImage(img, use_color_jitter=False, image_size_dict=cfg.IMAGESIZE_DICT, img_norm_info=cfg.IMAGE_NORMALIZE_INFO)
	else:
		raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
	input_img = input_img.unsqueeze(0).type(FloatTensor)
	gt_boxes = torch.FloatTensor([1, 1, 1, 1, 0]).unsqueeze(0).type(FloatTensor)
	img_info = torch.from_numpy(np.array([target_size[0], target_size[1], scale_factor])).unsqueeze(0).type(FloatTensor)
	num_gt_boxes = torch.FloatTensor([0]).unsqueeze(0).type(FloatTensor)
	with torch.no_grad():
		output = model(x=input_img, gt_boxes=gt_boxes, img_info=img_info, num_gt_boxes=num_gt_boxes)
	anchors = output[0].data.view(1, -1, 4)
	preds_cls = output[1].data
	preds_reg = output[2].data
	# parse the results
	preds_reg = preds_reg.view(-1, 4) * torch.FloatTensor(cfg.BBOX_NORMALIZE_STDS).type(FloatTensor) + torch.FloatTensor(cfg.BBOX_NORMALIZE_MEANS).type(FloatTensor)
	preds_reg = preds_reg.view(1, -1, 4)
	boxes_pred = BBoxFunctions.decodeBboxes(anchors, preds_reg)
	boxes_pred = BBoxFunctions.clipBoxes(boxes_pred, img_info.data)
	boxes_pred = boxes_pred.squeeze()
	scores = preds_cls.squeeze()
	thresh = 0.05
	for j in range(cfg.NUM_CLASSES-1):
		idxs = torch.nonzero(scores[:, j] > thresh).view(-1)
		if idxs.numel() > 0:
			cls_scores = scores[:, j][idxs]
			_, order = torch.sort(cls_scores, 0, True)
			cls_boxes = boxes_pred[idxs, :]
			cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
			cls_dets = cls_dets[order]
			cls_dets, _ = nms(cls_dets, args.nmsthresh)
			for cls_det in cls_dets:
				if cls_det[-1] > args.confthresh:
					x1, y1, x2, y2 = cls_det[:4]
					x1 = x1.item() / scale_factor
					x2 = x2.item() / scale_factor
					y1 = y1.item() / scale_factor
					y2 = y2.item() / scale_factor
					label = clsnames[j]
					logger_handle.info('Detect a %s in confidence %.4f...' % (label, cls_det[-1].item()))
					color = (0, 255, 0)
					draw = ImageDraw.Draw(img)
					draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill=color)
					font = ImageFont.truetype('libs/font.TTF', 25)
					draw.text((x1+5, y1), label, fill=color, font=font)
	img.save(os.path.join(cfg.TEST_BACKUPDIR, 'demo_output.jpg'))


'''run'''
if __name__ == '__main__':
	demo()