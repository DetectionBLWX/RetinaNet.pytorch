'''
Function:
	detect objects in one image
Author:
	Charles
'''
import torch
import warnings
import argparse
import numpy as np
from modules.utils.utils import *
from modules.utils.datasets import *
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
def test():
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
	cls_probs = output[0].data
	bbox_preds = output[1].data
	# parse the results



'''run'''
if __name__ == '__main__':
	demo()