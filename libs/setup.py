'''setup modified from mmdet'''
import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    define_macros = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError('CUDA is required to compile MMDetection!')
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


setup(
		name='retinanet',
		version='0.1.0',
		description='retinanet for object detection',
		classifiers=['License :: OSI Approved :: MIT License',
					 'Programming Language :: Python :: 3',
					 'Intended Audience :: Developers',
					 'Operating System :: OS Independent'],
		author='Charles',
		author_email='charlesjzc@qq.com',
		url='https://github.com/DetectionBLWX/RetinaNet.pytorch',
		license='MIT',
		include_package_data=True,
		packages=find_packages(),
		ext_modules=[
						make_cuda_ext(
										name='deform_conv_cuda', 
										module='dcn',
										sources=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu']
									),
						make_cuda_ext(
										name='deform_pool_cuda', 
										module='dcn',
										sources=['src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu']
									),
						make_cuda_ext(
										name='sigmoid_focal_loss_cuda', 
										module='sigmoid_focal_loss',
										sources=['src/sigmoid_focal_loss.cpp', 'src/sigmoid_focal_loss_cuda.cu']
									),
						make_cuda_ext(
										name='nms_cpu', 
										module='nms',
										sources=['src/nms_cpu.cpp']
									),
						make_cuda_ext(
										name='nms_cuda', 
										module='nms',
										sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']
									),
					],
		cmdclass={'build_ext': BuildExtension},
		zip_safe=False
	)