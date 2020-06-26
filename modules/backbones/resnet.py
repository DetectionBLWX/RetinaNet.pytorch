'''
Function:
	Feature Pyramid Network of ResNets
Author:
	Charles
'''
import torchvision


'''resnet from torchvision==0.4.0'''
def ResNets(resnet_type, pretrained=False):
	if resnet_type == 'resnet18':
		model = torchvision.models.resnet18(pretrained=pretrained)
	elif resnet_type == 'resnet34':
		model = torchvision.models.resnet34(pretrained=pretrained)
	elif resnet_type == 'resnet50':
		model = torchvision.models.resnet50(pretrained=pretrained)
	elif resnet_type == 'resnet101':
		model = torchvision.models.resnet101(pretrained=pretrained)
	elif resnet_type == 'resnet152':
		model = torchvision.models.resnet152(pretrained=pretrained)
	else:
		raise ValueError('Unsupport resnet_type <%s>...' % resnet_type)
	return model