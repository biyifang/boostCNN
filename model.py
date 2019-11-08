import torch
import torch.nn as nn
import numpy as np

class oneCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(oneCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=32, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=4))
		self.classifier = nn.Sequential(
			#nn.Dropout(),
			nn.Linear(16*12*12, num_classes),
			#nn.ReLU(inplace=True),
			#nn.Dropout(),
			#nn.Linear(4096, 4096),
			#nn.ReLU(inplace=True),
			#nn.Linear(4096, num_classes),
		)
		self.mse = nn.MSELoss()
	def forward(self, x, label=None, temperature=None):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		if label is not None:
			loss = torch.sum(label*nn.functional.log_softmax(x/temperature,-1), dim=1).mean()
			return -1.0*loss
		else:
			return x

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
			print(g)
			print(w)
			l = input('l')
			return self.mse(g, w)
		else:
			return g
		#data already with correct w/label
	def line_search(self, f, g, data):
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
		self.alpha.append((temp1 + temp2)/2)
	def predict(self, x, k):
		pred = next(self.weak_learners.parameters())
		pred = pred.new_zeros(x.size(0), self.num_classes)
		for i,net in enumerate(self.weak_learners):
			if i <= k:
				pred += net.forward(x) * self.alpha[i]*self.gamma
		#_, index = torch.max(pred, 0)
		return pred









		






