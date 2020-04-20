'''
Function:
	utils for distributed training, refer to mmdet and mmcv.
Author:
	Charles
'''
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


'''initialize distribution'''
def initializeDistribution(launcher='pytorch', backend='nccl', **kwargs):
	if mp.get_start_method(allow_none=True) is None:
		mp.set_start_method('spawn')
	if launcher == 'pytorch':
		rank = int(os.environ['RANK'])
		num_gpus = torch.cuda.device_count()
		torch.cuda.set_device(rank % num_gpus)
		dist.init_process_group(backend=backend, **kwargs)
	else:
		raise ValueError('Unsupport initializeDistribution.launcher <%s>...' % launcher)