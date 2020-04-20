'''
Function:
	builder for building dataloader
Author:
	Charles
'''
import torch
import platform
import numpy as np
from ..distribution import *


'''see https://github.com/pytorch/pytorch/issues/973'''
if platform.system() != 'Windows':
	import resource
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	hard_limit = rlimit[1]
	soft_limit = min(4096, hard_limit)
	resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


'''build dataloader for non-distributed training'''
def buildNonDistributedDataloader(dataset, cfg, mode, **kwargs):
	assert mode in ['TRAIN', 'TEST']
	if mode == 'TRAIN':
		sampler = cfg['sampler'](dataset.img_ratios, cfg['batch_size'])
		dataloader = torch.utils.data.DataLoader(dataset, 
												 batch_size=cfg['batch_size'], 
												 sampler=sampler, 
												 num_workers=cfg['num_workers'], 
												 collate_fn=cfg['collate_fn'], 
												 pin_memory=cfg['pin_memory'])
	else:
		dataloader = torch.utils.data.DataLoader(dataset,
												 batch_size=cfg['batch_size'],
												 num_workers=cfg['num_workers'],
												 shuffle=cfg['shuffle'],
												 pin_memory=cfg['pin_memory'])
	return dataloader


'''build dataloader for distributed training'''
def buildDistributedDataloader(dataset, cfg, mode, **kwargs):
	assert mode in ['TRAIN']
	rank, world_size = getDistributionInfo()
	sampler = cfg['sampler'](dataset.img_ratios, cfg['num_imgs_per_gpu'], world_size, rank)
	dataloader = torch.utils.data.DataLoader(dataset,
											 batch_size=cfg['num_imgs_per_gpu'],
											 sampler=sampler,
											 num_workers=cfg['num_workers_per_gpu'],
											 collate_fn=cfg['collate_fn'],
											 pin_memory=cfg['pin_memory'])
	return dataloader


'''build dataloader for training'''
def buildDataloader(dataset, cfg, mode, is_distribution=True, **kwargs):
	# distributed training
	if is_distribution:
		return buildDistributedDataloader(dataset, cfg, mode, **kwargs)
	# non-distributed training
	else:
		return buildNonDistributedDataloader(dataset, cfg, mode, **kwargs)