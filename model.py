import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import math
from torchvision.models.mobilenet import MobileNetV2



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class resNet18(nn.Module):
    def __init__(self, numClasses=10):
        super(resNet18, self).__init__()
        self.resNet18 = resnet18(num_classes=10)
    def forward(self, x, label=None, temperature=None, if_student = True):
        x = self.resNet18(x)
        if not if_student:
            return x
        if label is not None:
            loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x/temperature,-1), dim=1).mean()
            return -1.0*loss
        else:
            return nn.functional.softmax(x,-1)

class MobileNet_V2(nn.Module):
    def __init__(self, numClasses=10):
        super(MobileNet_V2, self).__init__()
        self.model = MobileNetV2(num_classes=10)
    def forward(self, x, label=None, temperature=None, if_student = True):
        x = self.model(x)
        if not if_student:
            return x
        if label is not None:
            loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x/temperature,-1), dim=1).mean()
            return -1.0*loss
        else:
            return nn.functional.softmax(x,-1)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetv2(nn.Module):
    def __init__(self, n_class=10, input_size=224, width_mult=1.):
        super(MobileNetv2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x, label=None, temperature=None, if_student = True):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        if not if_student:
            return x
        if label is not None:
            loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x/temperature,-1), dim=1).mean()
            return -1.0*loss
        else:
            return nn.functional.softmax(x,-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class oneCNN(nn.Module):
#MobileNet
    def __init__(self, num_classes=10):
        super(oneCNN, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 10)
        self.features_1 = nn.Sequential(
        #2/1-layer kernel=32 stride=4
            #nn.Conv2d(3, 16, kernel_size=32, stride=4, padding=2),
            #nn.Conv2d(3, 16, kernel_size=16, stride=4, padding=2),
            #nn.BatchNorm2d(16),
            #nn.Dropout(p=0.2),
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
            #nn.Sigmoid(),
            #nn.MaxPool2d(kernel_size=3, stride=2))
        self.features_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
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
        self.features_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7, stride=1))

            #nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=2),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=2),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=1))

            #nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=2),
            #nn.BatchNorm2d(16),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=1))
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            #2-layers
            #nn.Dropout(0.2),
            #nn.Linear(4*3*3, num_classes),
            #nn.Linear(4*5*5, num_classes),
            #3-layers
            nn.Linear(1024, num_classes),
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
        #print(x.size())
        #x_1 = self.features_1(x)
        #print(x_1.size())
        #x_f = torch.flatten(x_1, 1)
        #print('size 1')
        #print(x_f.size())
        #print('size 2')
        #print(self.res.weight.size())
        #x_res = self.res(x_f)
        #print('res done')
        #x_1 = torch.flatten(x_1, 1)
        #print(x_1.size())
        #x_1 = self.classifier(x_1 + x_res)
        #x_1 = self.classifier(x_1)

        x = self.model(x)
        x = x.view(-1, 1024)
        x_1 = self.fc(x)
        if not if_student:
            return x_1
        if label is not None:
            loss = torch.sum(nn.functional.softmax(label, -1)*nn.functional.log_softmax(x_1/temperature,-1), dim=1).mean()
            return -1.0*loss
        else:
            return nn.functional.softmax(x_1,-1)




class oneCNN_two(nn.Module):
    def __init__(self, num_classes=10):
        super(oneCNN_two, self).__init__()
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
            nn.Linear(32*3*3, num_classes),
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
        self.res = nn.Linear(128*19*19, 32*3*3)
        self.mse = nn.MSELoss()
    def forward(self, x, label=None, temperature=None, if_student = True):
        x_1 = self.features_1(x)
        x_f = torch.flatten(x_1, 1)
        #print(x_f.size())
        x_res = self.res(x_f)
        x_1 = self.features_2(x_1)
        x_1 = torch.flatten(x_1, 1)
        #print(x_1.size())
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
                        #weight[j] = 0.0
                weight[label] = 1.0 * (self.num_classes - 1)
                #weight[label] = 1.0
        else:
            alpha = self.alpha[iteration-1]
            for i,( (_,label) ,(weight,)) in enumerate( zip(data,weight_data)):
                temp_sum = 0.0
                for j in range(self.num_classes):
                    if j != label:
                        temp = - torch.exp(-1.0/2*self.gamma*alpha*(g[i][label] - g[i][j]))*weight[j]
                        weight[j] = temp
                        temp_sum += temp
                weight[label] = - temp_sum
    def forward(self, x, w, iteration, loss=True):
        g = self.weak_learners[iteration](x, if_student=False)
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
            return loss/len(label)
        seg = (np.sqrt(5) + 1)/2
        error = 1000
        while error >= merror:
            temp1 = upper - (upper - lower)/seg
            temp2 = lower + (upper - lower)/seg
            loss_temp1 = obj(f + temp1 * g, label, num_classes)
            loss_temp2 = obj(f + temp2 * g, label, num_classes)
            if loss_temp1 < loss_temp2:
                upper = temp2
            else:
                lower = temp1
            error = np.abs(loss_temp1 - loss_temp2)
        #self.alpha.append((temp1 + temp2)/2) plane
        self.alpha.append((temp1 + temp2)/(2*gamma))
    def predict(self, x, k):
        pred = next(self.weak_learners.parameters())
        pred = pred.new_zeros(x.size(0), self.num_classes).cuda()
        for i,net in enumerate(self.weak_learners):
            net.cuda()
            if i == 0:
                pred += net.forward(x, if_student=False)
            elif i <= k:
                pred += net.forward(x, if_student=False) * self.alpha[i]*self.gamma
            net.cpu()
        #_, index = torch.max(pred, 0)
        return pred









        






