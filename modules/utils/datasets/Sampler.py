'''
Function:
	define some sample methods.
Author:
	Charles
'''
import torch
import numpy as np
from ..distribution import *


'''group sampler'''
class GroupSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, img_ratios, batch_size, **kwargs):
		self.img_ratios = img_ratios
		self.batch_size = batch_size
		# divide images into two groups
		self.group_flags = np.array(img_ratios) >= 1
		self.group_sizes = np.bincount(self.group_flags)
		# calculate total sample times
		self.total_sample_times = 0
		for i, size in enumerate(self.group_sizes):
			self.total_sample_times += int(np.ceil(size / batch_size)) * batch_size
	'''iter'''
	def __iter__(self):
		indices = []
		for i, size in enumerate(self.group_sizes):
			if size > 0:
				indice = np.where(self.group_flags == i)[0]
				assert len(indice) == size
				np.random.shuffle(indice)
				num_extra = int(np.ceil(size / self.batch_size)) * self.batch_size - len(indice)
				indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
				indices.append(indice)
		indices = np.concatenate(indices)
		indices = [indices[i*self.batch_size: (i+1)*self.batch_size] for i in np.random.permutation(range(len(indices) // self.batch_size))]
		indices = np.concatenate(indices)
		indices = indices.astype(np.int64).tolist()
		assert len(indices) == self.total_sample_times
		return iter(indices)
	'''len'''
	def __len__(self):
		return self.total_sample_times