'''
Function:
	define data parallel
Author:
	Charles
'''
import torch
from torch.nn.parallel._functions import Scatter as OrigScatter


'''scatter inputs to target gpus'''
def scatter(inputs, target_gpus, dim=0):
	def scatterMap(obj):
		if isinstance(obj, torch.Tensor):
			return OrigScatter.apply(target_gpus, None, dim, obj)
		if isinstance(obj, tuple) and len(obj) > 0:
			return list(zip(*map(scatterMap, obj)))
		if isinstance(obj, list) and len(obj) > 0:
			return list(map(list, zip(*map(scatterMap, obj))))
		if isinstance(obj, dict) and len(obj) > 0:
			return list(map(type(obj), zip(*map(scatterMap, obj.items()))))
	try:
		return scatterMap(inputs)
	finally:
		scatterMap = None


'''scatter with support for kwargs dictionary'''
def scatterKwargs(inputs, kwargs, target_gpus, dim=0):
	inputs = scatter(inputs, target_gpus, dim) if inputs else []
	kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
	if len(inputs) < len(kwargs):
		inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
	elif len(kwargs) < len(inputs):
		kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
	inputs = tuple(inputs)
	kwargs = tuple(kwargs)
	return inputs, kwargs


'''non-distributed data parallel'''
class NonDistributedDataParallel(torch.nn.parallel.DataParallel):
	def scatter(self, inputs, kwargs, device_ids):
		return scatterKwargs(inputs, kwargs, device_ids, dim=self.dim)


'''distributed data parallel'''
class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
	def scatter(self, inputs, kwargs, device_ids):
		return scatterKwargs(inputs, kwargs, device_ids, dim=self.dim)