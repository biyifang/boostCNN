import torch
import torch.nn as nn
import numpy as np
import math

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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
        self.classifier = nn.ModuleList([
            nn.Dropout(),
            #change 256*6*6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)])

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for it in self.classifier:
            x = it(x)
        return x

    def forward(self, x, label=None, temperature=None, if_student = True):
        x = self._forward_impl(x)
        if not if_student:
            return x
        if label is not None:
            loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x/temperature,-1), dim=1).mean()
            return -1.0*loss
        else:
            return nn.functional.softmax(x,-1)

    def get_size(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x.size()[1]


class GBM(nn.Module):
    def __init__(self, num_iter,num_class, shrink_param, model_list=None):
        super(GBM, self).__init__()
        if not model_list:
            self.weak_learners = nn.ModuleList([oneCNN() for _ in range(num_iter)]) 
        else:
            self.weak_learners = nn.ModuleList(model_list)
        self.num_classes = num_class
        self.mse = nn.MSELoss()
        self.kl = nn.CrossEntropyLoss()
        self.alpha = [0.0 for _ in range(num_iter)]
        self.gamma = shrink_param
        self.subgrid = {}
    def weight_fun(self, data, weight_data, iteration, g):
        #data = TensorDataset(x,label,weight)
        if iteration == 0:
            for i, ( (_, label),  (weight,)) in enumerate(zip(data,weight_data) ):
                #print(weight)
                weight[:] = torch.ones_like(weight)*(-1.0)
                #for j in range(self.num_classes):
                #    if j != label:
                #        weight[j] = - 1.0
                        #weight[j] = 0.0
                weight[label] = 1.0 * (self.num_classes - 1)
                #weight[label] = 1.0
        else:
            alpha = self.alpha[iteration-1]
            for i,( (_,label) ,(weight,)) in enumerate( zip(data,weight_data)):
                temp_sum = 0.0
                temp = torch.exp(-1.0/2*self.gamma*alpha*(g[i][label] - g[i]))*weight
                temp[label] = 0.0
                temp[label] = torch.sum(temp) * (-1)
                weight[:] = temp
                #for j in range(self.num_classes):
                #    if j != label:
                #        temp = torch.exp(-1.0/2*self.gamma*alpha*(g[i][label] - g[i][j]))*weight[j]
                #        weight[j] = temp
                #        temp_sum += temp
                #weight[label] = - temp_sum
    def forward(self, x, w, iteration, loss=True, loss_type='mse'):
        g = self.weak_learners[iteration](x, if_student=False)
        if loss:
            if loss_type == 'mse':
                return self.mse(g, w)
            else:
                return self.kl(g, torch.argmax(w, dim=1))
        else:
            #return g
            return nn.functional.softmax(g,-1)
        #data already with correct w/label
    #def line_search(self, f, g, data): plane
    def line_search(self, f, g, data, gamma):
        #data = TensorDataset(f,g,label) sequntial data
        lower = 0.0
        upper = 1.0
        merror = 1e-4
        label = [ it[1]  for it in data ]
        num_classes = self.num_classes
        def obj(pred, label, num_classes):
            loss = 0.0
            for i in range(len(label)):
                loss += torch.sum(torch.exp(-1.0/2*(torch.ones(num_classes)*pred[i, label[i]] - pred[i,:]))) - 1
            return loss/len(label)
        seg = (np.sqrt(5) + 1)/2
        error = 1000
        while error >= merror:
            temp1 = upper - (upper - lower)/seg
            temp2 = lower + (upper - lower)/seg
            '''
            print('temp1')
            print(temp1)
            print(temp2)
            print(f)
            print(g)
            print(label)
            '''
            loss_temp1 = obj(f + temp1 * g, label, num_classes)
            loss_temp2 = obj(f + temp2 * g, label, num_classes)
            '''
            print('loss')
            print(loss_temp1)
            print(loss_temp2)
            '''
            if loss_temp1 < loss_temp2:
                upper = temp2
            else:
                lower = temp1
            error = np.abs(loss_temp1 - loss_temp2)
        #self.alpha.append((temp1 + temp2)/2) plane
        #return (temp1 + temp2)/(2*gamma)
        return (temp1 + temp2)/2
    def predict(self, x, k):
        pred = next(self.weak_learners.parameters())
        pred = pred.new_zeros(x.size(0), self.num_classes).cuda()
        for i,net in enumerate(self.weak_learners):
            net.cuda()
            if i == 0:
                #x_start, y_start, x_end, y_end, stepsize = self.subgrid[i]
                #pred += net.forward(x[:,:,x_start:x_end+1:stepsize, y_start:y_end+1:stepsize], if_student=False)
                
                x_axis, y_axis = self.subgrid[i]
                pred += net.forward(x[:,:, x_axis,:][:,:,:,y_axis], if_student=False)
            elif i <= k:
                #x_start, y_start, x_end, y_end, stepsize = self.subgrid[i]
                #pred += net.forward(x[:,:,x_start:x_end+1:stepsize, y_start:y_end+1:stepsize], if_student=False) * self.alpha[i]*self.gamma

                x_axis, y_axis = self.subgrid[i]
                pred += net.forward(x[:,:, x_axis,:][:,:,:,y_axis], if_student=False) * self.alpha[i]*self.gamma
            net.cpu()
        #_, index = torch.max(pred, 0)
        pred.cpu()
        return pred
        '''
        previous_prob = prob.cuda()
        self.weak_learners[k].cuda()
        previous_prob += self.weak_learners[k].forward(x, if_student=False) * self.alpha[k]*self.gamma
        self.weak_learners[k].cpu()
        return previous_prob.cpu()
        '''
    def predict_fast(self, x, i):
        net = self.weak_learners[i]
        if i == 0:
            #x_start, y_start, x_end, y_end, stepsize = self.subgrid[i]
            #pred += net.forward(x[:,:,x_start:x_end+1:stepsize, y_start:y_end+1:stepsize], if_student=False)
                
            x_axis, y_axis = self.subgrid[i]
            pred = net.forward(x[:,:, x_axis,:][:,:,:,y_axis], if_student=False)
        else:
            #x_start, y_start, x_end, y_end, stepsize = self.subgrid[i]
            #pred += net.forward(x[:,:,x_start:x_end+1:stepsize, y_start:y_end+1:stepsize], if_student=False) * self.alpha[i]*self.gamma

            x_axis, y_axis = self.subgrid[i]
            pred = net.forward(x[:,:, x_axis,:][:,:,:,y_axis], if_student=False) * self.alpha[i]*self.gamma
        return pred

