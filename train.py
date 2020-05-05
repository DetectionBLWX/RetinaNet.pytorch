'''
Function:
	train the model
Author:
	Charles
'''
import os
import torch
import warnings
import argparse
import torch.distributed as dist
from modules.utils import *
from modules.optimizer import *
from modules.RetinaNet import RetinanetFPNResNets
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
	parser = argparse.ArgumentParser(description='RetinaNet')
	parser.add_argument('--datasetname', dest='datasetname', help='dataset for training.', default='', type=str, required=True)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for training.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str)
	parser.add_argument('--local_rank', dest='local_rank', help='local rank', default=0, type=int)
	args = parser.parse_args()
	if 'LOCAL_RANK' not in os.environ:
		os.environ['LOCAL_RANK'] = str(args.local_rank)
	return args


'''train the model'''
def train():
	# prepare base things
	args = parseArgs()
	cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
	checkDir(cfg.TRAIN_BACKUPDIR)
	logger_handle = Logger(cfg.TRAIN_LOGFILE)
	use_cuda = torch.cuda.is_available()
	is_multi_gpus, is_distributed_training = cfg.IS_MULTI_GPUS, cfg.IS_DISTRIBUTED_TRAINING
	if is_multi_gpus: assert use_cuda
	if is_multi_gpus and is_distributed_training:
		initializeDistribution(launcher=cfg.INIT_DISTRIBUTION_SET['launcher'], backend=cfg.INIT_DISTRIBUTION_SET['backend'])
	# prepare dataset
	if args.datasetname == 'coco':
		dataset = COCODataset(rootdir=cfg.DATASET_ROOT_DIR, image_size_dict=cfg.IMAGESIZE_DICT, max_num_gt_boxes=cfg.MAX_NUM_GT_BOXES, use_color_jitter=cfg.USE_COLOR_JITTER, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, mode='TRAIN', datasettype='train2017')
		if cfg.IS_DISTRIBUTED_TRAINING:
			build_dataloader_set = cfg.BUILD_DATALOADER_SET['distributed']
			build_dataloader_set.update({'collate_fn': COCODataset.paddingCollateFn, 'sampler': DistributedGroupSampler})
		else:
			build_dataloader_set = cfg.BUILD_DATALOADER_SET['non_distributed']
			build_dataloader_set.update({'collate_fn': COCODataset.paddingCollateFn, 'sampler': GroupSampler})
		dataloader = buildDataloader(dataset, cfg=build_dataloader_set, mode='TRAIN', is_distribution=cfg.IS_DISTRIBUTED_TRAINING)
	else:
		raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
	# prepare model
	if args.backbonename.find('resnet') != -1:
		model = RetinanetFPNResNets(mode='TRAIN', cfg=cfg, logger_handle=logger_handle)
	else:
		raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
	start_epoch = 1
	end_epoch = cfg.MAX_EPOCHS
	if use_cuda:
		model = model.cuda()
	# prepare optimizer
	learning_rate_idx = 0
	if cfg.IS_USE_WARMUP:
		learning_rate = cfg.LEARNING_RATES[learning_rate_idx] / 3
	else:
		learning_rate = cfg.LEARNING_RATES[learning_rate_idx]
	if cfg.OPTIMIZER_SET['type'] == 'sgd':
		optimizer_set = cfg.OPTIMIZER_SET['sgd']
		optimizer_set.update({'learning_rate': learning_rate})
		optimizer = SGDBuilder(model, optimizer_set, True)
	else:
		raise ValueError('Unsupport optimizer <%s> now...' % cfg.OPTIMIZER_SET['type'])
	# check checkpoints path
	if args.checkpointspath:
		checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
		model.load_state_dict(checkpoints['model'])
		optimizer.load_state_dict(checkpoints['optimizer'])
		start_epoch = checkpoints['epoch'] + 1
		for epoch in range(1, start_epoch):
			if epoch in cfg.LR_ADJUST_EPOCHS:
				learning_rate_idx += 1
	# data parallel
	if is_multi_gpus and is_distributed_training:
		model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=False)
	elif is_multi_gpus:
		model = NonDistributedDataParallel(model, device_ids=range(torch.cuda.device_count()))
	# print config
	if args.local_rank == 0:
		logger_handle.info('Dataset used: %s, Number of images: %s' % (args.datasetname, len(dataset)))
		logger_handle.info('Backbone used: %s' % args.backbonename)
		logger_handle.info('Checkpoints used: %s' % args.checkpointspath)
		logger_handle.info('Config file used: %s' % cfg_file_path)
	# train
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	for epoch in range(start_epoch, end_epoch+1):
		# --set train mode
		if is_multi_gpus:
			model.module.setTrain()
		else:
			model.setTrain()
		# --set epcoh for dataloader
		if is_multi_gpus and is_distributed_training:
			dataloader.sampler.setEpoch(epoch)
		# --adjust learning rate
		if epoch in cfg.LR_ADJUST_EPOCHS:
			learning_rate_idx += 1
			adjustLearningRate(optimizer=optimizer, target_lr=cfg.LEARNING_RATES[learning_rate_idx], logger_handle=logger_handle)
		# --log info
		logger_handle.info('Start epoch %s, learning rate is %s...' % (epoch, cfg.LEARNING_RATES[learning_rate_idx]))
		# --train epoch
		for batch_idx, samples in enumerate(dataloader):
			if (epoch == 1) and (cfg.IS_USE_WARMUP) and (batch_idx <= cfg.NUM_WARMUP_STEPS):
				assert learning_rate_idx == 0, 'BUGS may exist...'
				target_lr = cfg.LEARNING_RATES[learning_rate_idx] / 3
				target_lr += (cfg.LEARNING_RATES[learning_rate_idx] - cfg.LEARNING_RATES[learning_rate_idx] / 3) * batch_idx / cfg.NUM_WARMUP_STEPS
				adjustLearningRate(optimizer=optimizer, target_lr=target_lr)
			optimizer.zero_grad()
			img_ids, imgs, gt_boxes, img_info, num_gt_boxes = samples
			output = model(x=imgs.type(FloatTensor), gt_boxes=gt_boxes.type(FloatTensor), img_info=img_info.type(FloatTensor), num_gt_boxes=num_gt_boxes.type(FloatTensor))
			anchors, preds_cls, preds_reg, loss_cls, loss_reg = output
			loss = loss_cls.mean() + loss_reg.mean()
			if is_multi_gpus and is_distributed_training:
				rank, world_size = getDistributionInfo()
				loss_cls_log = loss_cls.data.clone()
				dist.all_reduce(loss_cls_log.div_(dist.get_world_size()))
				loss_cls_log = loss_cls_log.item()
				loss_reg_log = loss_reg.data.clone()
				dist.all_reduce(loss_reg_log.div_(dist.get_world_size()))
				loss_reg_log = loss_reg_log.item()
				loss_log = loss.data.clone()
				dist.all_reduce(loss_log.div_(dist.get_world_size()))
				loss_log = loss_log.item()
			else:
				rank = 0
				loss_cls_log = loss_cls.mean().item()
				loss_reg_log = loss_reg.mean().item()
				loss_log = loss.item()
			if rank == 0:
				logger_handle.info('[EPOCH]: %s/%s, [BATCH]: %s/%s, [LEARNING_RATE]: %s, [DATASET]: %s \n\t [LOSS]: loss_cls %.4f, loss_reg %.4f, total %.4f' % \
									(epoch, end_epoch, (batch_idx+1), len(dataloader), cfg.LEARNING_RATES[learning_rate_idx], args.datasetname, loss_cls_log, loss_reg_log, loss_log))
			loss.backward()
			clipGradients(model.parameters(), cfg.GRAD_CLIP_MAX_NORM, cfg.GRAD_CLIP_NORM_TYPE)
			optimizer.step()
		# --save model
		if (args.local_rank == 0) and ((epoch % cfg.SAVE_INTERVAL == 0) or (epoch == end_epoch)):
			state_dict = {'epoch': epoch,
						  'model': model.module.state_dict() if is_multi_gpus else model.state_dict(),
						  'optimizer': optimizer.state_dict()}
			savepath = os.path.join(cfg.TRAIN_BACKUPDIR, 'epoch_%s.pth' % epoch)
			saveCheckpoints(state_dict, savepath, logger_handle)


'''run'''
if __name__ == '__main__':
	train()