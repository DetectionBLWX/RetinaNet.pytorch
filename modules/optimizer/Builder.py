'''
Function:
	define the optimizer builders
Author:
	Charles
'''
import torch.optim as optim


'''sgd builder'''
def SGDBuilder(model, cfg, is_filter_params=True, **kwargs):
	if is_filter_params:
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
							  lr=cfg['learning_rate'], 
							  momentum=cfg['momentum'], 
							  weight_decay=cfg['weight_decay'])
	else:
		optimizer = optim.SGD(model.parameters(), 
							  lr=cfg['learning_rate'], 
							  momentum=cfg['momentum'], 
							  weight_decay=cfg['weight_decay'])
	return optimizer