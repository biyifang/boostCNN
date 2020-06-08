import argparse
import os
import random
import shutil
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm,trange
from subgrid_model import oneCNN
from subgrid_model import oneCNN_two
from subgrid_model import MobileNetv2
from subgrid_model import MobileNet_V2
from subgrid_model import resNet18
from subgrid_model import GBM
from torch.utils.data import TensorDataset

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('--data', metavar='DIR', default='/Users/biyifang/Desktop/research/AllState/experiment', type=str,
					help='path to dataset')
parser.add_argument('--model_save', metavar='MS', default='', type=str,
					help='path to student model')
parser.add_argument('--teacher_model_save', metavar='MS', default='', type=str,
					help='path to teacher model')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--bs_epochs', default=10, type=int, metavar='N',
					help='number of total epochs to run in boosting CNN')
parser.add_argument('--subgrid_epochs', default=10, type=int, metavar='N',
					help='number of total epochs to run in subgrid boosting CNN')
parser.add_argument('--CNN_one', default=5, type=int, metavar='N',
					help='the kernel size of CNN layer 1')
parser.add_argument('--CNN_two', default=5, type=int, metavar='N',
					help='the kernel size of CNN layer 2')
parser.add_argument('--CNN_three', default=3, type=int, metavar='N',
					help='the kernel size of CNN layer 3')
parser.add_argument('--num_class', default=10, type=int, metavar='NoC',
					help='number of class')
parser.add_argument('--num_boost_iter', default=50, type=int, metavar='N',
					help='number of boosting iterations')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')#default:0.1
parser.add_argument('-gradient_acc', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr_dis', '--learning-rate-dis', default=0.001, type=float,
					metavar='LRdis', help='learning rate for distillation', dest='lr_dis')
parser.add_argument('--lr_boost', '--learning-rate-boost', default=0.000001, type=float,
					metavar='LRboost', help='learning rate for boosting', dest='lr_boost')
parser.add_argument('--lr_sub', default=0.000001, type=float,
					metavar='LRsubgrid', help='learning rate for subgrid training')
parser.add_argument('--temperature', '--temperature', default=3.0, type=float,
					metavar='temperature', help='temperature for softmax', dest='temperature')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--boost_shrink', default=0.99, type=float, metavar='S',
					help='boosting shrinkage parameter')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10000, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')

best_acc1 = 0


def main():
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	# create model
	if args.pretrained:
		print("=> using pre-trained model '{}'".format(args.arch))
		model = models.__dict__[args.arch](pretrained=True)
	else:
		print("=> creating model '{}'".format(args.arch))
		model = models.__dict__[args.arch]()
		#model = models.resnet18(num_classes=10)
		#model = torch.load('initial_model_' + args.model_save)
		model = torch.load('teacher_model_resnet18')
		model.cuda()


	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss()
	#criterion = nn.MSELoss()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code
	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	train_dataset = datasets.CIFAR10(args.data, train=True, transform=transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=True)
	


	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None
		train_sampler_seq = torch.utils.data.SequentialSampler(train_dataset)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	train_loader_seq = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, sampler=train_sampler_seq)

	
	val_dataset = datasets.CIFAR10(args.data, train=False, transform=transforms.Compose([
			#transforms.RandomResizedCrop(224),
			transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
			#transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=False)
	
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	image_list = []
	label_list = []
	with torch.no_grad():
		for i, (image, label) in enumerate(train_loader_seq):
			image = image.cuda()
			image_embedding = model(image, if_student=False)
			image_list.append(image_embedding.cpu())
			label_list.append(label)
	train_embedding = torch.cat(image_list).numpy()
	train_label = torch.cat(label_list).numpy()

	np.save('train_embedding', train_embedding)
	np.save('train_label', train_label)

	image_list = []
	label_list = []
	with torch.no_grad():
		for i, (image, label) in enumerate(val_loader):
			image = image.cuda()
			image_embedding = model(image, if_student=False)
			image_list.append(image_embedding.cpu())
			label_list.append(label)
	val_embedding = torch.cat(image_list).numpy()
	val_label = torch.cat(label_list).numpy()
	np.save('val_embedding', val_embedding)
	np.save('val_label', val_label)


if __name__ == '__main__':
	main()



	



