import torch
import torch.nn as nn
from itertools import product
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

import sys 
sys.path.append('../model')

from downsample import Downsample
        


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureExtractor(nn.Module):

    def __init__(self):
        
        self.inplanes = 64
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                                         
                                         
        
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
                                        Downsample(filt_size=3, stride=2, channels=64)])
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [Downsample(filt_size=3, stride=stride, channels=self.inplanes),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1), nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)
        
    
        
    def do_forward(self, x) : 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        return x       
                    
    def forward(self, x):
        
        if self.training : 
            x = self.do_forward(x)
        else : ## if it is evaluation mode, switch to no grad, to accelerate 
            with torch.no_grad() : 
                x = self.do_forward(x)
        
        return x

        
        
class CorrNeigh(nn.Module):
    def __init__(self, kernelSize):
        super(CorrNeigh, self).__init__()
        assert kernelSize % 2 == 1 
        self.kernelSize = kernelSize
        self.paddingSize = kernelSize // 2
        self.padding = torch.nn.ZeroPad2d(self.paddingSize) 
        
    
    def do_forward(self, x, y):
        
        ## x, y should be normalized
        n, c, w, h = x.size()
        coef = []
        y = self.padding(y)
        ## coef is the feature similarity between (i,j) and (i-r, j-r) with -kernel < r < +kernel 
        for i,j in product(range(self.kernelSize), range(self.kernelSize)) : 
            coef.append( torch.sum(x * y.narrow(2, i, w).narrow(3, j, h), dim=1, keepdim=True) )
        coef = torch.cat(coef, dim=1)
        
        return coef
    
    def forward(self, x, y):
        
        if self.training : 
            coef = self.do_forward(x, y)
        
        else : ## if it is evaluation mode, switch to no grad, to accelerate 
            with torch.no_grad() : 
                coef = self.do_forward(x, y)
        
        return coef
   
    



        
class NetFlowCoarse(nn.Module):
    def __init__(self, kernelSize):
        super(NetFlowCoarse, self).__init__()
        assert kernelSize % 2 == 1
        
        self.conv1 =  conv3x3(kernelSize * kernelSize, 512)
        
        self.bn1 =  nn.BatchNorm2d(512, eps=1e-05)
        self.relu =  nn.ReLU(inplace=True)
        
        self.conv2 =  conv3x3(512, 256)
        self.bn2 =  nn.BatchNorm2d(256, eps=1e-05)
        
        self.conv3 =  conv3x3(256, 128)
        self.bn3 =  nn.BatchNorm2d(128, eps=1e-05)
        
        self.conv4 =  conv3x3(128, kernelSize * kernelSize)
        
        
        
        self.kernelSize = kernelSize
        self.paddingSize = kernelSize // 2
        
        self.gridY = torch.arange(-self.paddingSize, self.paddingSize + 1).view(1, 1, -1, 1).expand(1, 1, self.kernelSize,  self.kernelSize).contiguous().view(1, -1, 1, 1).type(torch.FloatTensor)
        self.gridX = torch.arange(-self.paddingSize, self.paddingSize + 1).view(1, 1, 1, -1).expand(1, 1, self.kernelSize,  self.kernelSize).contiguous().view(1, -1, 1, 1).type(torch.FloatTensor)
        self.softmax = torch.nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def cuda(self):
        super().cuda()
        self.gridX, self.gridY = self.gridX.cuda(), self.gridY.cuda()

        
    def do_forward(self, coef, up8X):
        ## x, y should be normalized
        n, c, w, h = coef.size()
        x = self.conv1(coef)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        
        ## scale the similarity term and do softmax (self.heat is learnt as well) 
        x = self.softmax(x)
        
        ## flow 
        flowX = torch.sum(x * self.gridX, dim=1, keepdim=True) / h * 2
        flowY = torch.sum(x * self.gridY, dim=1, keepdim=True) / w * 2
        flow = torch.cat((flowX, flowY), dim=1)
        flow = F.upsample_bilinear(flow, size=None, scale_factor=8) if up8X else flow
        
        return flow
        
        
    def forward(self, coef, up8X=True):
        
        if self.training : 
            flow = self.do_forward(coef, up8X)
        
        else : ## if it is evaluation mode, switch to no grad, to accelerate 
            with torch.no_grad() : 
                flow = self.do_forward(coef, up8X)
        
        
        return flow
        
        


class NetMatchability(nn.Module):
    def __init__(self, kernelSize):
        super(NetMatchability, self).__init__()
        
        self.conv1 =  conv3x3(kernelSize * kernelSize, 512)
        
        self.bn1 =  nn.BatchNorm2d(512, eps=1e-05)
        self.relu =  nn.ReLU(inplace=True)
        
        self.conv2 =  conv3x3(512, 256)
        self.bn2 =  nn.BatchNorm2d(256, eps=1e-05)
        
        self.conv3 =  conv3x3(256, 128)
        self.bn3 =  nn.BatchNorm2d(128, eps=1e-05)
        
        self.conv4 =  conv3x3(128, 1)
        
        
        self.sigmoid =  nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
               
        ## make the initial matchability to 0.5 
        nn.init.normal_(self.conv4.weight, mean=0.0, std=0.0001)
        
        
        
    def do_forward(self, feat, up8X):
        ## x, y should be normalized
        n, c, w, h = feat.size()
        x = self.conv1(feat)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        
        
        x = self.sigmoid(x)
        
        x = F.upsample_bilinear(x, size=None, scale_factor=8) if up8X else x
        
        return x
        
    def forward(self, feat, up8X=True):
        
        if self.training : 
            matchability = self.do_forward(feat, up8X)
        
        else : ## if it is evaluation mode, switch to no grad, to accelerate 
            with torch.no_grad() : 
                matchability = self.do_forward(feat, up8X)
        
        return matchability
        
    
        

## L1 pixel wise loss, maskMargin and margin are useless here
def L1PixelWise(I1Warp, I2, match2, margin = None, maskMargin = None, ssim=None) : 
    return torch.sum( match2 * torch.abs(I1Warp - I2) ) / torch.sum(match2) / 3
    
## Take the central part to estimate pixel transformation
def L1PixelShift(I1Warp, I2, match2, margin, maskMargin, ssim=None) : 
    
    n, _, w, h = I1Warp.size()
    gridSize = w - 2 * margin 
    I1Grid = I1Warp.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    I2Grid = I2.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    MGrid = maskMargin.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    
    I1Grid = I1Grid.contiguous().view(I1Grid.size(0), 3, -1) # N, C, W* H
    I2Grid = I2Grid.contiguous().view(I2Grid.size(0), 3, -1) # N, C, W* H
    MGrid = MGrid.contiguous().view(MGrid.size(0), 1, -1).expand(MGrid.size(0), 3, gridSize * gridSize) # N, 3, W* H
    MGridSum = torch.sum(MGrid, dim = 2, keepdim=True) # N, 3, 1
    
    
    s = torch.cuda.FloatTensor(I1Grid.size(0), I1Grid.size(1), 1).fill_(1)
    d = torch.cuda.FloatTensor(I1Grid.size(0), I1Grid.size(1), 1).fill_(0)
    
    
    with torch.no_grad() : 
        
        m1 = (I1Grid * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1 
        m2 = (I2Grid * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1
        v1 = (((I1Grid - m1) ** 2) * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1 
        v2 = (((I2Grid - m2) ** 2) * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1
        mask = (v1 * v2 != 0) 
        s[mask] = (v2[mask] / v1[mask]) ** 0.5
        d = m2 - s * m1
    
    I1T = (torch.clamp(I1Warp * s.unsqueeze(3) + d.unsqueeze(3), min = 0., max = 1.))
    
    
    return torch.sum( match2 * torch.abs(I1T - I2) ) / (torch.sum(match2) + 0.1) / 3
    

## Take the central part to estimate pixel transformation
def SSIMPixelShift(I1Warp, I2, match2, margin, maskMargin, ssim) : 
    
    n, _, w, h = I1Warp.size()
    gridSize = w - 2 * margin 
    I1Grid = I1Warp.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    I2Grid = I2.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    MGrid = maskMargin.narrow(2, margin, gridSize).narrow(3, margin, gridSize)
    
    I1Grid = I1Grid.contiguous().view(I1Grid.size(0), 3, -1) # N, C, W* H
    I2Grid = I2Grid.contiguous().view(I2Grid.size(0), 3, -1) # N, C, W* H
    MGrid = MGrid.contiguous().view(MGrid.size(0), 1, -1).expand(MGrid.size(0), 3, gridSize * gridSize) # N, 3, W* H
    MGridSum = torch.sum(MGrid, dim = 2, keepdim=True) # N, 3, 1
    
    
    s = torch.cuda.FloatTensor(I1Grid.size(0), I1Grid.size(1), 1).fill_(1)
    d = torch.cuda.FloatTensor(I1Grid.size(0), I1Grid.size(1), 1).fill_(0)
    
    
    with torch.no_grad() : 
        
        m1 = (I1Grid * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1 
        m2 = (I2Grid * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1
        v1 = (((I1Grid - m1) ** 2) * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1 
        v2 = (((I2Grid - m2) ** 2) * MGrid).sum(dim=2, keepdim=True) / (MGridSum + 1e-7) # N, C, 1
        mask = (v1 * v2 != 0) 
        s[mask] = (v2[mask] / v1[mask]) ** 0.5
        d = m2 - s * m1
    
    I1T = (torch.clamp(I1Warp * s.unsqueeze(3) + d.unsqueeze(3), min = 0., max = 1.))
    
    
    return ssim(I1T, I2, match2)
    
## Take the central part to estimate pixel transformation
def SSIM(I1Warp, I2, match2, margin, maskMargin, ssim) :     
    return ssim(I1Warp, I2, match2)
    

def predFlowCoarse(corrKernel21, NetFlowCoarse, grid, up8X = True) : 
    
    flowCoarse = NetFlowCoarse(corrKernel21, up8X) ## output is with dimension B, 2, W, H
    b, _, w, h = flowCoarse.size()
    flowGrad = flowCoarse.narrow(2, 1, w-1).narrow(3, 1, h-1) - flowCoarse.narrow(2, 0, w-1).narrow(3, 0, h-1)
    flowGrad = torch.norm(flowGrad, dim=1, keepdim=True)
    flowCoarse = flowCoarse.permute(0, 2, 3, 1)
    flowCoarse = torch.clamp(flowCoarse + grid, min=-1, max=1)
    
    return flowGrad, flowCoarse
    
def predFlowCoarseNoGrad(corrKernel21, NetFlowCoarse, grid, up8X = True) : 
    
    flowCoarse = NetFlowCoarse(corrKernel21, up8X) ## output is with dimension B, 2, W, H
    b, _, w, h = flowCoarse.size()
    
    flowCoarse = flowCoarse.permute(0, 2, 3, 1)
    flowCoarse = torch.clamp(flowCoarse + grid, min=-1, max=1)
        
    return flowCoarse

    
def predMatchability(corrKernel21, NetMatchability, up8X = True) : 
    
    matchability = NetMatchability(corrKernel21, up8X)
    
    return matchability

