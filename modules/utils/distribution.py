'''
Function:
	utils for distributed training, refer to mmdet and mmcv.
Author:
	Charles
'''
import torch
import torch.distributed as dist


'''get distribution info'''
def getDistributionInfo():
	if torch.__version__ < '1.0':
		is_initialized = dist._initialized
	else:
		if dist.is_available():
			is_initialized = dist.is_initialized()
		else:
			is_initialized = False
	if is_initialized:
		rank = dist.get_rank()
		world_size = dist.get_world_size()
	else:
		rank = 0
		world_size = 1
	return rank, world_size