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
	def __init__(self, img_ratios, num_samples_per_gpu, **kwargs):
		self.img_ratios = img_ratios
		self.num_samples_per_gpu = num_samples_per_gpu
		# divide images into two groups
		self.group_flags = np.array(img_ratios) >= 1
		self.group_sizes = np.bincount(self.group_flags)
		# calculate total sample times
		self.total_sample_times = 0
		for i, size in enumerate(self.group_sizes):
			self.num_samples += int(np.ceil(size / num_samples_per_gpu)) * num_samples_per_gpu
	'''iter'''
	def __iter__(self):
		indices = []
		for i, size in enumerate(self.group_sizes):
			if size > 0:
				indice = np.where(self.group_flags == i)[0]
				assert len(indice) == size
				np.random.shuffle(indice)
				num_extra = int(np.ceil(size / self.num_samples_per_gpu)) * self.num_samples_per_gpu - len(indice)
				indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
				indices.append(indice)
		indices = np.concatenate(indices)
		indices = [indices[i*self.num_samples_per_gpu: (i+1)*self.num_samples_per_gpu] for i in np.random.permutation(range(len(indices) // self.num_samples_per_gpu))]
		indices = np.concatenate(indices)
		indices = indices.astype(np.int64).tolist()
		assert len(indices) == self.num_samples
		return iter(indices)
	'''len'''
	def __len__(self):
		return self.total_sample_times


'''distributed group sampler'''
class DistributedGroupSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, img_ratios, num_samples_per_gpu, num_replicas, rank, epoch=0, **kwargs):
		self.img_ratios = img_ratios
		self.num_samples_per_gpu = num_samples_per_gpu
		self.num_replicas = num_replicas
		self.rank = rank
		self.epoch = epoch
		# divide images into two groups
		self.group_flags = np.array(img_ratios) >= 1
		self.group_sizes = np.bincount(self.group_flags)
		# calculate total sample times
		self.total_sample_times = 0
		for i, size in enumerate(self.group_sizes):
			self.total_sample_times += int(np.ceil(size * 1.0 / num_samples_per_gpu / num_replicas)) * num_samples_per_gpu
		# calculate total size
		self.total_size = self.total_sample_times * self.num_replicas
	'''iter'''
	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch)
		indices = []
		for i, size in enumerate(self.group_sizes):
			if size > 0:
				indice = np.where(self.group_flags == i)[0]
				assert len(indice) == size
				indice = indice[list(torch.randperm(int(size), generator=g))].tolist()
				num_extra = int(np.ceil(size * 1.0 / self.num_samples_per_gpu / self.num_replicas)) * self.num_samples_per_gpu * self.num_replicas - len(indice)
				tmp = indice.copy()
				for _ in range(num_extra // size):
					indice.extend(tmp)
				indice.extend(tmp[: num_extra%size])
				indices.extend(indice)
		assert len(indices) == self.total_size
		indices = [indices[j] for i in list(torch.randperm(len(indices) // self.num_samples_per_gpu, generator=g)) for j in range(i * self.num_samples_per_gpu, (i + 1) * self.num_samples_per_gpu)]
		offset = self.total_sample_times * self.rank
		indices = indices[offset: offset+self.total_sample_times]
		assert len(indices) == self.total_sample_times
		return iter(indices)
	'''len'''
	def __len__(self):
		return self.total_sample_times
	'''set epoch'''
	def setEpoch(self, epoch):
		self.epoch = epoch