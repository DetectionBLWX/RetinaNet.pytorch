'''
Function:
	builder for building dataloader
Author:
	Charles
'''
import torch
import platform
import numpy as np


'''build dataloader for training'''
def buildDataloader(dataset, cfg, mode, **kwargs):
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