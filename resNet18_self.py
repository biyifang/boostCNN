import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import math
from torchvision.models.mobilenet import MobileNetV2

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x.size()[1]


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class GBM(nn.Module):
    def __init__(self, num_iter, shrink_param, model_list=None):
        super(GBM, self).__init__()
        if not model_list:
            self.weak_learners = nn.ModuleList([oneCNN() for _ in range(num_iter)]) 
        else:
            self.weak_learners = nn.ModuleList(model_list)
        self.num_classes = 100
        self.mse = nn.MSELoss()
        self.alpha = [0.0 for _ in range(num_iter)]
        self.gamma = shrink_param
        self.subgrid = {}
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
            #return g
            return self.weak_learners[iteration](x)
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
        return (temp1 + temp2)/(2*gamma)
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

