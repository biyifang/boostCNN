import argparse
import os
import random
import shutil
import time
import warnings
import copy

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
from resNet18_self import ResNet
from resNet18_self import GBM
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
		model = ResNet()
		#model = mobilenet_v2()
		#model = MobileNet_V2()
		#model = oneCNN()
		model.cuda()

	"""
	if args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()
	"""

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
	#Normalization for MNIST
	#normalize = transforms.Normalize(mean=[0.485], std=[0.229])

	'''
	train_dataset = datasets.SVHN(args.data, split='train', transform=transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=True)
	'''
	train_dataset = datasets.CIFAR10(args.data, train=True, transform=transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=True)

	'''
	train_dataset = datasets.MNIST(args.data, train=True, transform=transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	]), target_transform=None, download=True)
	'''
	weight = torch.zeros(len(train_dataset), args.num_class)
	weight_dataset = torch.utils.data.TensorDataset(weight)


	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None
		train_sampler_seq = torch.utils.data.SequentialSampler(train_dataset)
		weight_sampler = torch.utils.data.SequentialSampler(weight_dataset )

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	train_loader_seq = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, sampler=train_sampler_seq)
	weight_loader = torch.utils.data.DataLoader(
		 weight_dataset, batch_size=args.batch_size, sampler=weight_sampler)

	'''
	val_dataset = datasets.SVHN(args.data, split='test', transform=transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=True)
	'''
	val_dataset = datasets.CIFAR10(args.data, train=False, transform=transforms.Compose([
			#transforms.RandomResizedCrop(224),
			transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
			#transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=False)
	
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
	probability = torch.zeros(len(val_dataset), args.num_class)
	probability_dataset = torch.utils.data.TensorDataset(probability)
	probability_sampler = torch.utils.data.SequentialSampler(probability_dataset )
	probability_loader = torch.utils.data.DataLoader(
		 probability_dataset, batch_size=args.batch_size, sampler=probability_sampler)
	'''
	val_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data, train=False, transform=transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=False), batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
	'''

	'''
	if args.evaluate:
		validate(val_loader, model, criterion, args)
		return
	'''

	'''
	#step one: find a good teacher model
	if args.teacher_model_save:
		model = torch.load('teacher_model_' + args.teacher_model_save)
	else:
		for epoch in trange(args.start_epoch, args.epochs):
			if args.distributed:
				train_sampler.set_epoch(epoch)
			adjust_learning_rate(optimizer, epoch, args)

			# train for one epoch
			train(train_loader, model, criterion, optimizer, epoch, args)
			print('Iteration: ' + str(epoch) + '\n')

			# evaluate on validation set
			acc1 = validate(val_loader, model, criterion, args)

			# remember best acc@1 and save checkpoint
			is_best = acc1 > best_acc1
			best_acc1 = max(acc1, best_acc1)
			if acc1 == best_acc1:
				_, new_predict = validate(train_loader, model, criterion, args, True)
				new_predict = torch.cat(new_predict)
				predict_dataset = torch.utils.data.TensorDataset(new_predict)
				predict_sampler = torch.utils.data.SequentialSampler(predict_dataset )
				predict_loader = torch.utils.data.DataLoader(
					predict_dataset, batch_size=args.batch_size, sampler=predict_sampler)
				model.cpu()
				torch.save(model, 'SVHN_teacher_model_resnet18')
				model.cuda()


			if not args.multiprocessing_distributed or (args.multiprocessing_distributed
					and args.rank % ngpus_per_node == 0):
				save_checkpoint({
					'epoch': epoch + 1,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer' : optimizer.state_dict(),
				}, is_best)
		#_, new_predict = validate(train_loader, model, criterion, args, True)
		#new_predict = torch.cat(new_predict)
		#predict_dataset = torch.utils.data.TensorDataset(new_predict)
		#predict_sampler = torch.utils.data.SequentialSampler(predict_dataset )
		#predict_loader = torch.utils.data.DataLoader(
		#     predict_dataset, batch_size=args.batch_size, sampler=predict_sampler)
		print(best_acc1)
		#l = input('l')
	model.cpu()
	'''
	

	'''
	#if have teacher model, no need to run step one
	model = torch.load('SVHN_teacher_model_resnet18')
	_, new_predict = validate(train_loader, model, criterion, args, True)
	new_predict = torch.cat(new_predict)
	predict_dataset = torch.utils.data.TensorDataset(new_predict)
	predict_sampler = torch.utils.data.SequentialSampler(predict_dataset )
	predict_loader = torch.utils.data.DataLoader(
		predict_dataset, batch_size=args.batch_size, sampler=predict_sampler)
	model.cpu()
	'''


	'''
	# one-layer CNN training
	inter_media_1 = kernel_fun(224, args.CNN_one, 4, 2)
	inter_media_two = maxpool_fun(inter_media_1, 3, 2)
	inter_media_3 = kernel_fun(inter_media_two, args.CNN_two, 2, 2)
	inter_media_4 = maxpool_fun(inter_media_3, 2, 2)
	inter_media_5 = kernel_fun(inter_media_4, args.CNN_three, 2, 2)
	inter_media_six = maxpool_fun(inter_media_5, 2,1)
	model_2 = oneCNN_two(args.CNN_one, args.CNN_two, args.CNN_three, inter_media_two, inter_media_six)
	#model_2 = torch.hub.load('pytorch/vision:v0.5.0','mobilenet_v2', pretrained=True)
	model_2.cuda()
	#optimizer = torch.optim.SGD(model_2.parameters(), args.lr_dis, momentum=args.momentum, weight_decay=args.weight_decay)
	optimizer = torch.optim.Adam(model_2.parameters(),args.lr_dis)
	model_2.train()
	acc2 = 0.0
	print('start step 2')
	optimizer.zero_grad()
	for epoch in trange(args.epochs):
		lo = 0.0
		top1 = AverageMeter('Acc@1', ':6.2f')
		for i, ( (images, target), (label,)) in enumerate( tqdm(zip(train_loader_seq , predict_loader)) ):
			images = images.cuda()
			label = label.cuda()
			target = target.cuda()
			loss = model_2(images, label, args.temperature)

			output = model_2(images)
			acc1, _ = accuracy(output, target, topk=(1, 5))
			top1.update(acc1[0], images.size(0))

			lo += loss.data
			# compute gradient and do SGD step
			loss.backward()
			if i%args.gradient_acc == 0:
				optimizer.step()
				optimizer.zero_grad()
		if top1.avg > acc2:
			acc2 = top1.avg
			model_2.cpu()
			torch.save(model_2, 'SVHN_initial_model_'+ args.model_save)
			model_2.cuda()
		print('iteration ' + str(epoch) + ': ' + str(lo.data) + '\t' + 'accuracy: ' + str(top1.avg)+'\n')
	print('oneCNN optimization done')
	optimizer.zero_grad()
	#l = input('l')
	#model = None

	# boosted CNN
	model_2.cpu()
	'''

	model.cpu()
	output_file = open('out.txt','w')
	



	#Create module for GBM
	#model_2 = torch.load('SVHN_initial_model_' + args.model_save)
	#model_list = [copy.deepcopy(model_2) for _ in range(args.num_boost_iter)]
	#model_2 = oneCNN()
	#model_2 = mobilenet_v2()
	#model_2 = resNet18()
	#model_2 = MobileNet_V2()
	#model_2_1 = oneCNN_two(CNN_one, CNN_two, CNN_three, inter_media_two, inter_media_six)

	model_list = [copy.deepcopy(model)]
	model_3 = GBM(args.num_boost_iter, args.boost_shrink, model_list)
	model_3.cpu()
	model_3.train()
	optimizer_list = [torch.optim.SGD(it.parameters(), args.lr_boost,
								momentum=args.momentum,
								weight_decay=args.weight_decay) for it in model_3.weak_learners]
	
	g = None
	f = torch.zeros(len(train_dataset), args.num_class)

	#Train GBM
	for k in trange(0,args.num_boost_iter):
		if args.distributed:
			train_sampler.set_epoch(epoch)

		'''
		if k >= 1:
			model_list = model_list + [copy.deepcopy(model_3.weak_learners[k-1])]
			alpha = model_3.alpha
			#model_list = [ copy.deepcopy(model_2_1) for _ in range(args.num_boost_iter)]
			model_3 = GBM(args.num_boost_iter, args.boost_shrink, model_list)
			model_3.alpha = alpha
			model_3.cpu()
			model_3.train()
			optimizer_list = [torch.optim.SGD(it.parameters(), args.lr_boost,
								momentum=args.momentum,
								weight_decay=args.weight_decay) for it in model_3.weak_learners]
		'''

		# train for one epoch
		if k == 0:
			f, g = train_boost(train_loader_seq,weight_loader,weight_dataset, train_dataset, model_3, optimizer_list, k, f, g, args)
			#model_3.subgrid[0] = (0,0,223,223,1)
			temp = [i for i in range(224)]
			model_3.subgrid[0] = (temp, temp)

			grad_value = find_grad(train_dataset, weight_dataset, model_3, optimizer_list, k, args)

			x_axis_opt = temp
			y_axis_opt = temp
			acc1 = validate_boost(val_loader, model_3, criterion, args, k)
			#(a,b,x)	
		else:
			#train_boost(train_loader_seq,weight_loader,weight_dataset, train_dataset, model_3, optimizer_list, k, f, g, args)
			#model_3.subgrid[k] = (0,0,223,223,1)
			temp = [i for i in range(224)]
			model_3.subgrid[k] = (temp, temp)
			#find gradient
			#grad_value = find_grad(train_dataset, weight_dataset, model_3, optimizer_list, k, args)

			#update certain pixels
			grad_value_temp = find_grad(train_dataset, weight_dataset, model_3, optimizer_list, k, args)
			grad_value[x_axis_opt,:][:,y_axis_opt] = grad_value_temp[x_axis_opt,:][:,y_axis_opt]

			#acc_temp = validate_boost(val_loader, model_3, criterion, args, k)
			#print('iteration: ' + str(k) + '   accuracy :' + str(acc_temp))
		
		'''	
		# initialize the weight for the next weak learner
		model_list = model_list + [copy.deepcopy(model_3.weak_learners[k])]
		alpha = model_3.alpha
		subgrid_map = model_3.subgrid
		model_3 = GBM(args.num_boost_iter, args.boost_shrink, model_list)
		model_3.alpha = alpha
		model_3.subgrid = subgrid_map
		model_3.cpu()
		model_3.train()
		optimizer_list = [torch.optim.SGD(it.parameters(), args.lr_boost,
								momentum=args.momentum,
								weight_decay=args.weight_decay) for it in model_3.weak_learners]
		'''


		if k > 0:
			#set_grad_to_false(model_3.weak_learners[k].features_1)
			#set_grad_to_false(model_3.weak_learners[k].features_2)
			grad_opt = 0.0

			'''
			for x in range(2, 7):
			#for x in trange(223,224):
				for a in range(20):
					for b in range(20):
						if a <= b:
							y_axis = [i for i in range(b, 224, x)]
							x_axis = [i for i in range(a,224,x)][:len(y_axis)]
						else:
							x_axis = [i for i in range(a, 224, x)]
							y_axis = [i for i in range(b,224,x)][:len(x_axis)]
						
						grad_temp = torch.mean(grad_value[x_axis, y_axis])
						#print(grad_temp)
						if grad_temp > grad_opt:
							x_start_opt = x_axis[0]
							x_end_opt = x_axis[-1]
							y_start_opt = y_axis[0]
							y_end_opt = y_axis[-1]
							stepsize_opt = x
							grad_opt = grad_temp
							#print(x_start_opt)
			model_3.subgrid[k] = (x_start_opt,y_start_opt, x_end_opt, y_end_opt, stepsize_opt)
			print(model_3.subgrid[k])
			input_size = (x_end_opt - x_start_opt)/stepsize_opt + 1
			'''

			for x in range(180, 202):
			#134, 180,202
				'''
				index = [i for i in range(224)]
				del index[::x]
				images = images[:,:,index,:]
				images = images[:,:,:,index]
				'''
				for a in range(10):
					for b in range(10):
						x_axis = sorted(random.sample(range(a,224), x))
						y_axis = sorted(random.sample(range(b,224), x))
						
						grad_temp = torch.mean(grad_value[x_axis,:][:, y_axis])
						#print(grad_temp)
						if grad_temp > grad_opt:
							x_axis_opt = x_axis
							y_axis_opt = y_axis
							x_opt = x
							a_opt = a
							b_opt = b
							grad_opt = grad_temp
							#print(x_start_opt)
			model_3.subgrid[k] = (x_axis_opt,y_axis_opt)

			print('a: ' + str(a_opt) + '\t' + 'b: '+ str(b_opt) + '\t' + 'x: ' + str(x_opt))
			#input_size = int((223 - max(a_opt, b_opt) + x_opt)/x_opt)

			'''
			input_size = x_opt
			inter_media_1_t = kernel_fun(input_size, args.CNN_one, 4, 2)
			inter_media_two_t = maxpool_fun(inter_media_1_t, 3, 2)
			inter_media_3_t = kernel_fun(inter_media_two_t, args.CNN_two, 2, 2)
			inter_media_4_t = maxpool_fun(inter_media_3_t, 2, 2)
			inter_media_5_t = kernel_fun(inter_media_4_t, args.CNN_three, 2, 2)
			inter_media_six_t = maxpool_fun(inter_media_5_t, 2,1)
			model_3.weak_learners[k].classifier = nn.Linear(32*inter_media_six_t*inter_media_six_t, args.num_class)
			model_3.weak_learners[k].res = nn.Linear(128*inter_media_two_t*inter_media_two_t, 32*inter_media_six_t*inter_media_six_t)
			optimizer_list[k] = torch.optim.Adam(model_3.weak_learners[k].parameters(), args.lr_sub, 
					weight_decay=args.weight_decay)
			'''
			with torch.no_grad():
				for i, (images, target) in enumerate(tqdm(train_loader_seq)):
					images = images[:,:, x_axis,:][:,:,:,y_axis].cuda()
					print(next(model_3.weak_learners[k].parameters()))
					new_size = model_3.weak_learners[k].get_size(images)
					break
			model_3.weak_learners[k].fc = nn.Linear(new_size, 10)
			optimizer_list[k] = torch.optim.Adam(model_3.weak_learners[k].parameters(), args.lr_sub, 
					weight_decay=args.weight_decay)
			f, g = subgrid_train(train_loader_seq, train_dataset, weight_loader, model_3, optimizer_list, k, f, g,args)
			print('end subgrid train')
			validate_boost(train_loader_seq, model_3, criterion, args, k)
			acc1 = validate_boost(val_loader, model_3, criterion, args, k)

			'''
			#update gradient
			grad_value_temp = find_grad(train_dataset, weight_dataset, model_3, optimizer_list, k, args)
			grad_value[x_axis_opt,:][:,y_axis_opt] = grad_value_temp
			'''

		# initialize the weight for the next weak learner
		model_list = model_list + [copy.deepcopy(model_3.weak_learners[k])]
		alpha = model_3.alpha
		subgrid_map = model_3.subgrid
		model_3 = GBM(args.num_boost_iter, args.boost_shrink, model_list)
		model_3.alpha = alpha
		model_3.subgrid = subgrid_map
		model_3.weak_learners[k+1].fc = model_3.weak_learners[0].fc
		model_3.cpu()
		model_3.train()
		optimizer_list = [torch.optim.SGD(it.parameters(), args.lr_boost,
								momentum=args.momentum,
								weight_decay=args.weight_decay) for it in model_3.weak_learners]





		# fix the current model
		# train: image[:,:,:168,:168]

		# evaluate on validation set
		#acc1 = validate_boost(val_loader, model_3, criterion, args, k, probability_loader)
		output_file.write('Iteration {} * Acc@1 {:5.5f} '.format(k, acc1))

		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)

		'''
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			save_checkpoint({
				'iteration': k + 1,
				'arch': args.arch,
				'state_dict': model_3.state_dict(),
				'best_acc1': best_acc1,
				'optimizer' : optimizer.state_dict(),
			}, is_best)
		'''
	output_file.close()


def train(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (images, target) in enumerate(tqdm(train_loader)):
		# measure data loading time
		data_time.update(time.time() - end)

		images = images.cuda()
		target = target.cuda()

		# compute output
		output = model(images,if_student=False)
		#output = model(images)
		output = output/args.temperature
		#target_1 = nn.functional.one_hot(target, num_classes = 10).float()
		loss = criterion(output, target)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images.size(0))
		top1.update(acc1[0], images.size(0))
		top5.update(acc5[0], images.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		'''
		if i % args.print_freq == 0:
			progress.display(i)
		'''


def validate(val_loader, model, criterion, args, Flag = False):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()
	if Flag:
		new_label = []

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(val_loader):
			images = images.cuda()
			target = target.cuda()

			# compute output
			output = model(images, if_student=False)
			#output = model(images)
			#output = output/args.temperature
			if Flag:
				new_label.append(output.data.cpu())
			#target_1 = nn.functional.one_hot(target, num_classes = 10).float()
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			'''
			if i % args.print_freq == 0:
				progress.display(i)
			'''

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	if Flag:
		return top1.avg,new_label
	else:
		return top1.avg
def set_grad_to_false(model):
	for p in model.parameters():
		p.required_grad = False

def find_grad(train_dataset, weight_dataset, model, optimizer_list, k, args):
	x_axis, y_axis = model.subgrid[k]

	optimizer = optimizer_list[k]
	model.weak_learners[k].cuda()
	model.eval()
	train_sampler = torch.utils.data.SequentialSampler(train_dataset)
	weight_sampler = torch.utils.data.SequentialSampler(weight_dataset)
	train_loader_seq = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
	weight_loader = torch.utils.data.DataLoader(weight_dataset, sampler=weight_sampler, batch_size=1)
	grad_input = None
	model.weak_learners[k].zero_grad()
	for i, ((images, target),(weight,)) in enumerate( tqdm(zip(train_loader_seq , weight_loader)) ):
		#images = images.cuda()
		images = images[:,:, x_axis,:][:,:,:,y_axis].cuda()
		images.requires_grad = 	True
		target = target.cuda()
		#print(target)
		weight = weight.cuda()
		output = model(images, weight, k)
		output.backward()
		if grad_input is None:
			grad_input = torch.abs(images.grad.data).sum(1) + 0.0
		else:
			grad_input += torch.abs(images.grad.data).sum(1) + 0.0
		model.weak_learners[k].zero_grad()

	return grad_input[0]/len(train_dataset)

def subgrid_train(train_loader_seq, train_dataset, weight_loader, model, optimizer_list, k, f, g,args):
	#x_start, y_start, x_end, y_end, stepsize = model.subgrid[k]
	x_axis, y_axis = model.subgrid[k]
	optimizer = optimizer_list[k]
	model.weak_learners[k].cuda()
	model.train()
	optimizer.zero_grad()
	for epoch in trange(args.subgrid_epochs):
		for i, ( (images, _), (weight,)) in enumerate( tqdm(zip(train_loader_seq , weight_loader)) ):
			#images = images[:,:, x_start:x_end+1:stepsize, y_start:y_end+1:stepsize].cuda()
			images = images[:,:, x_axis,:][:,:,:,y_axis].cuda()
			weight = weight.cuda()
			#set_grad_to_false(model.weak_learners[k].features_1)
			#set_grad_to_false(model.weak_learners[k].features_2)

			# compute output
			loss = model(images, weight, k)      

			# measure accuracy and record loss
			#losses.update(loss.item(), images.size(0))

			# compute gradient and do SGD step
			loss.backward()
			if i%args.gradient_acc == 0:
				optimizer.step()
				optimizer.zero_grad()
	optimizer.zero_grad()
	g = []
	model.eval()
	for i, ( (images, _), (weight,)) in enumerate(zip(train_loader_seq , weight_loader) ):
		#images = images[:,:, x_start:x_end+1:stepsize, y_start:y_end+1:stepsize].cuda()
		images = images[:,:, x_axis,:][:,:,:,y_axis].cuda()
		weight = weight.cuda()
		with torch.no_grad():
			g.append(model(images, weight, k, False).detach())
	g = torch.cat(g, 0).cpu()
	# model.line_search(f, g, train_dataset) plane
	model.alpha[k] = model.line_search(f, g, train_dataset, model.gamma)
	f = f + model.gamma*model.alpha[k] * g
	print(model.alpha)
	model.weak_learners[k].cpu()
	return f, g


def train_boost( train_loader_seq, weight_loader, weight_dataset, train_dataset, model, optimizer_list, k, f, g, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	optimizer = optimizer_list[k]
	model.weak_learners[k].cuda()
	progress = ProgressMeter(
		len(train_loader_seq),
		[batch_time, data_time, losses, top1, top5],
		prefix="Iteration: [{}]".format(k))

	# switch to train mode
	model.train()

	end = time.time()

	model.weight_fun(train_dataset,weight_dataset, k, g)

	optimizer.zero_grad()
	if k == 0:
		bs_epochs = 0
	else:
		bs_epochs = args.bs_epochs
	for epoch in trange(bs_epochs):
		for i, ( (images, _), (weight,)) in enumerate( tqdm(zip(train_loader_seq , weight_loader)) ):
			# measure data loading time
			data_time.update(time.time() - end)

			images = images.cuda()
			weight = weight.cuda()


			# compute output
			loss = model(images, weight, k)      

			# measure accuracy and record loss
			losses.update(loss.item(), images.size(0))

			# compute gradient and do SGD step
			loss.backward()
			if i%args.gradient_acc == 0:
				optimizer.step()
				optimizer.zero_grad()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			#if (i+1) % args.print_freq == 0:
			#    progress.display(i)
	optimizer.zero_grad()

	g = []
	model.eval()
	for i, ( (images, _), (weight,)) in enumerate(zip(train_loader_seq , weight_loader) ):
		images = images.cuda()
		weight = weight.cuda()
		if i == 0:
			print(weight)
		with torch.no_grad():
			if i == 0:
				print(model(images, weight, k, False).detach())
			g.append(model(images, weight, k, False).detach())
	g = torch.cat(g, 0).cpu()
	if k == 0:
		model.alpha[0] = 1.0
		f = g
	else:
		model.alpha[k] = model.line_search(f, g, train_dataset, model.gamma)
		f = f + model.gamma*model.alpha[k] * g
	print(model.alpha)
	model.weak_learners[k].cpu()
	return f, g


def validate_boost(val_loader, model, criterion, args, k):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(val_loader):

			images = images.cuda()
			target = target.cuda()

			# compute output
			output = model.predict(images, k)
			output = output.cuda()
			#output = output/args.temperature
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.display(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

def mobilenet_v2():
    return MobileNetv2(width_mult=1)

def kernel_fun(orig_size, kern, stri, pad):
	return int((orig_size + 2*pad - (kern - 1) - 1)/stri + 1)

def maxpool_fun(orig_size, kern, stri):
	return int((orig_size - (kern - 1) - 1)/stri + 1)

class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	main()