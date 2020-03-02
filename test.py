'''
Function:
	test mAP
Author:
	Charles
'''
import json
import torch
import argparse
import warnings
import numpy as np
from modules.utils.utils import *
from modules.utils.datasets import *
from libs.nms.nms_wrapper import nms
from modules.RetinaNet import RetinanetFPNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings("ignore")


'''parse arguments in command line'''
def parseArgs():
	parser = argparse.ArgumentParser(description='RetinaNet')
	parser.add_argument('--datasetname', dest='datasetname', help='dataset for testing.', default='', type=str, required=True)
	parser.add_argument('--annfilepath', dest='annfilepath', help='used to specify annfilepath.', default='', type=str)
	parser.add_argument('--datasettype', dest='datasettype', help='used to specify datasettype.', default='val2017', type=str)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for testing.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str, required=True)
	parser.add_argument('--nmsthresh', dest='nmsthresh', help='thresh used in nms.', default=0.5, type=float)
	args = parser.parse_args()
	return args


'''test mAP'''
def test():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TEST_BACKUPDIR)
	logger_handle = Logger(cfg.TEST_LOGFILE)
	use_cuda = torch.cuda.is_available()
	clsnames = loadclsnames(cfg.CLSNAMESPATH)
	# prepare dataset
	if args.datasetname == 'coco':
		dataset = COCODataset(rootdir=cfg.DATASET_ROOT_DIR, image_size_dict=cfg.IMAGESIZE_DICT, max_num_gt_boxes=-1, use_color_jitter=False, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, mode='TEST', datasettype=args.datasettype, annfilepath=args.annfilepath)
	else:
		raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
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
	# test mAP
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	results = []
	img_ids = []
	for batch_idx, samples in enumerate(dataloader):
		logger_handle.info('detect %s/%s...' % (batch_idx+1, len(dataloader)))
		# --do detect
		img_id, img, w_ori, h_ori, gt_boxes, img_info, num_gt_boxes = samples
		img_id, w_ori, h_ori, scale_factor = int(img_id.item()), w_ori.item(), h_ori.item(), img_info[0][-1].item()
		img_ids.append(img_id)
		with torch.no_grad():
			output = model(x=img.type(FloatTensor), gt_boxes=gt_boxes.type(FloatTensor), img_info=img_info.type(FloatTensor), num_gt_boxes=num_gt_boxes.type(FloatTensor))
		cls_probs = output[0].data
		bbox_preds = output[1].data
		# --parse the results
		pass
	json.dump(results, open(cfg.TEST_BBOXES_SAVE_PATH, 'w'), indent=4)
	if args.datasettype in ['val2017']:
		dataset.doDetectionEval(img_ids, cfg.TEST_BBOXES_SAVE_PATH)


'''run'''
if __name__ == '__main__':
	test()