import torch
import torch.nn as nn
import numpy as np


class oneCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(oneCNN, self).__init__()
		self.features_1 = nn.Sequential(
		#2/1-layer kernel=32 stride=4
			#nn.Conv2d(3, 16, kernel_size=32, stride=4, padding=2),
			#nn.Conv2d(3, 16, kernel_size=16, stride=4, padding=2),
			#nn.BatchNorm2d(16),
			#nn.Dropout(p=0.2),
			nn.Conv2d(3, 128, kernel_size=16, stride=4, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			#nn.Sigmoid(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		'''
		self.features_2 = nn.Sequential(
			#nn.Conv2d(16, 4, kernel_size=8, stride=4, padding=2),
			nn.Dropout(p=0.2),
			nn.Conv2d(64, 4, kernel_size=8, stride=4, padding=2),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			#nn.Sigmoid(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		'''
		self.features_2 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=1))
		self.classifier = nn.Sequential(
			#nn.Dropout(),
			#2-layers
			#nn.Dropout(0.2),
			#nn.Linear(4*3*3, num_classes),
			#nn.Linear(4*5*5, num_classes),
			#3-layers
			nn.Linear(32*4*4, num_classes),
			#nn.ReLU(inplace=True),
			#nn.Dropout(),
			#nn.Linear(4096, 4096),
			#nn.ReLU(inplace=True),
			#nn.Linear(4096, num_classes),
		)
		#2-layers
		#self.res = nn.Linear(16*26*26, 4*3*3)
		#self.res = nn.Linear(64*26*26, 4*3*3)
		#3-layers
		self.res = nn.Linear(128*26*26, 32*4*4)
		self.mse = nn.MSELoss()
	def forward(self, x, label=None, temperature=None, if_student = True):
		x_1 = self.features_1(x)
		x_f = torch.flatten(x_1, 1)
		x_res = self.res(x_f)
		x_1 = self.features_2(x_1)
		x_1 = torch.flatten(x_1, 1)
		x_1 = self.classifier(x_1 + x_res)
		#x_1 = self.classifier(x_1)
		if not if_student:
			return x_1
		if label is not None:
			loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x_1/temperature,-1), dim=1).mean()
			return -1.0*loss
		else:
			return nn.functional.softmax(x_1,-1)




class oneCNN_two(nn.Module):
	def __init__(self, CNN_one, CNN_two, CNN_three, intermedia_1, intermida_2, num_classes=10,):
		super(oneCNN_two, self).__init__()
		self.features_1 = nn.Sequential(
		#2/1-layer kernel=32 stride=4
			#nn.Conv2d(3, 16, kernel_size=32, stride=4, padding=2),
			#nn.Conv2d(3, 16, kernel_size=16, stride=4, padding=2),
			#nn.BatchNorm2d(16),
			#nn.Dropout(p=0.2),
			nn.Conv2d(3, 128, kernel_size=CNN_one, stride=4, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			#nn.Sigmoid(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		'''
		self.features_2 = nn.Sequential(
			#nn.Conv2d(16, 4, kernel_size=8, stride=4, padding=2),
			nn.Dropout(p=0.2),
			nn.Conv2d(64, 4, kernel_size=8, stride=4, padding=2),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			#nn.Sigmoid(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		'''
		self.features_2 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=CNN_two, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 32, kernel_size=CNN_three, stride=2, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=1))
		self.classifier = nn.Sequential(
			#nn.Dropout(),
			#2-layers
			#nn.Dropout(0.2),
			#nn.Linear(4*3*3, num_classes),
			#nn.Linear(4*5*5, num_classes),
			#3-layers
			nn.Linear(32*intermida_2*intermida_2, num_classes),
			#nn.ReLU(inplace=True),
			#nn.Dropout(),
			#nn.Linear(4096, 4096),
			#nn.ReLU(inplace=True),
			#nn.Linear(4096, num_classes),
		)
		#2-layers
		#self.res = nn.Linear(16*26*26, 4*3*3)
		#self.res = nn.Linear(64*26*26, 4*3*3)
		#3-layers
		self.res = nn.Linear(128*intermedia_1*intermedia_1, 32*intermida_2*intermida_2)
		self.mse = nn.MSELoss()
	def forward(self, x, label=None, temperature=None, if_student = True):
		x_1 = self.features_1(x)
		x_f = torch.flatten(x_1, 1)
		print(x_f.size())
		x_res = self.res(x_f)
		x_1 = self.features_2(x_1)
		x_1 = torch.flatten(x_1, 1)
		print(x_1.size())
		x_1 = self.classifier(x_1 + x_res)
		#x_1 = self.classifier(x_1)
		if not if_student:
			return x_1
		if label is not None:
			loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x_1/temperature,-1), dim=1).mean()
			return -1.0*loss
		else:
			return nn.functional.softmax(x_1,-1)




'''
class oneCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(oneCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x, label=None, temperature=None, if_student = True):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x_1 = self.classifier(x)
		if not if_student:
			return x_1
		if label is not None:
			loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x_1/temperature,-1), dim=1).mean()
			return -1.0*loss
		else:
			return nn.functional.softmax(x_1,-1)
'''


class GBM(nn.Module):
	def __init__(self, num_iter, shrink_param, model_list=None):
		super(GBM, self).__init__()
		if not model_list:
			self.weak_learners = nn.ModuleList([oneCNN() for _ in range(num_iter)]) 
		else:
			self.weak_learners = nn.ModuleList(model_list)
		self.num_classes = 10
		self.mse = nn.MSELoss()
		self.alpha = []
		self.gamma = shrink_param
	def weight_fun(self, data, weight_data, iteration, g):
		#data = TensorDataset(x,label,weight)
		if iteration == 0:
			for i, ( (_, label),  (weight,)) in enumerate(zip(data,weight_data) ):
				#print(weight)
				for j in range(self.num_classes):
					if j != label:
						weight[j] = - 1.0
				weight[label] = 1.0 * (self.num_classes - 1)
		else:
			alpha = self.alpha[iteration-1]
			for i,( (_,label) ,(weight,)) in enumerate( zip(data,weight_data)):
				temp_sum = 0.0
				for j in range(self.num_classes):
					if j != label:
						temp = - torch.exp(-1.0/2*self.gamma*alpha*(g[i][label] - g[i][j])*weight[j])
						weight[j] = temp
						temp_sum += temp
				weight[label] = - temp_sum
	def forward(self, x, w, iteration, loss=True):
		g = self.weak_learners[iteration](x)
		if loss:
			return self.mse(g, w)
		else:
			return g
		#data already with correct w/label
	#def line_search(self, f, g, data): plane
	def line_search(self, f, g, data, gamma):
		#data = TensorDataset(f,g,label) sequntial data
		lower = 0.0
		upper = 1.0
		merror = 1e-5
		label = [ it[1]  for it in data ]
		num_classes = self.num_classes
		def obj(pred, label, num_classes):
			loss = 0.0
			for i in range(len(label)):
				loss += torch.sum(torch.exp(-1.0/2*(torch.ones(num_classes)*pred[i, label[i]] - pred[i,:]))) - 1
			return loss
		seg = (np.sqrt(5) - 1)/2
		error = 1000
		while error >= merror:
			temp1 = upper - seg*(upper - lower)
			temp2 = lower + seg*(upper - lower)
			loss_temp1 = obj(f + temp1 * g, label, num_classes)
			loss_temp2 = obj(f + temp2 * g, label, num_classes)
			if loss_temp1 > loss_temp2:
				upper = temp2
			else:
				lower = temp1
			error = torch.abs(loss_temp1 - loss_temp2)
		#self.alpha.append((temp1 + temp2)/2) plane
		self.alpha.append((temp1 + temp2)/(2*gamma))
	def predict(self, x, k):
		pred = next(self.weak_learners.parameters())
		pred = pred.new_zeros(x.size(0), self.num_classes).cuda()
		for i,net in enumerate(self.weak_learners):
			net.cuda()
			if i <= k:
				if i == 0:
					x_1 = x
				elif i == 1:
					x_1 = x[:, :, :168, :168]
				elif i == 2:
					x_1 = x[:, :, :168, 56:]
				elif i == 3:
					x_1 = x[:, :, 56:, :169]
				elif i == 4:
					x_1 = x[:, :, 56:, 56:]
				pred += net.forward(x_1) * self.alpha[i]*self.gamma
			net.cpu()
		#_, index = torch.max(pred, 0)
		return pred









		






