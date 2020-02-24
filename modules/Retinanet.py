'''
Function:
    define the Retinanet
Author:
    Charles
'''
import torch
import torch.nn as nn


'''base model for Retinanet'''
class Retinanet(nn.Module):
    def __init__(self, **kwargs):
        super(Retinanet, self).__init__()
    '''set bn eval'''
	@staticmethod
	def setBnEval(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.eval()