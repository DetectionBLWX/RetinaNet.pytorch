'''
Function:
    train the model
Author:
    Charles
'''
import torch
import warnings
import argparse
import torch.nn as nn
from modules.utils.utils import *
from modules.utils.datasets import *
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='RetinaNet')
    parser.add_argument('--datasetname', dest='datasetname', help='dataset for training.', default='', type=str, required=True)
    parser.add_argument('--backbonename', dest='backbonename', help='backbone network for training.', default='', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str)
	args = parser.parse_args()
    return args


'''train the model'''
def train():
    pass




'''run'''
if __name__ == '__main__':
    train()