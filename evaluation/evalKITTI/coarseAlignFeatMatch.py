import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
   
import torchvision.models as models
import pickle 

import pandas as pd
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")


sys.path.append('../../utils')
import outil

sys.path.append('../../segNet')
import segEval

sys.path.append('../../model')
from resnet50 import resnet50


from scipy.misc import imresize
from scipy import signal
## resize image according to the minsize, at the same time resize the x,y coordinate


class CoarseAlign:
    def __init__(self, nbScale, nbIter, tolerance, transform, minSize, segId, segFg, scaleR = 2, imageNet = True, segNet = True):
        
        ## nb iteration, tolerance, transform
        self.nbIter = nbIter
        self.tolerance = tolerance
        
        ## resnet 50 
        resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
        if imageNet : 
            resNetfeat = models.resnet50(pretrained=True)          
        else : 
            
            resNetfeat = resnet50()
            featPth = '../../model/pretrained/resnet50_moco.pth'
            param = torch.load(featPth)
            state_dict = {k.replace("module.", ""): v for k, v in param['model'].items()}
            msg = 'Loading pretrained model from {}'.format(featPth)
            print (msg)
            resNetfeat.load_state_dict( state_dict )
            
                
        resnet_module_list = [getattr(resNetfeat,l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index('layer3')
        self.net = torch.nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        self.net.cuda()
        self.net.eval()
        if segNet : 
            self.segNet = segEval.SegNet('../../model/pretrained/ade20k_resnet50dilated_encoder.pth', '../../model/pretrained/ade20k_resnet50dilated_decoder.pth', segId, segFg)
        ## preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.toTensor = transforms.ToTensor()
        self.preproc = transforms.Compose([transforms.ToTensor(), normalize,])
        
        if transform == 'Affine' :
            self.Transform = outil.Affine
            self.nbPoint = 3

        else : 
            self.Transform = outil.Homography
            self.nbPoint = 4
            
        self.strideNet = 16
        self.minSize = minSize
        
        if nbScale == 1 :
            self.scaleList = [1]
        
        else :
            self.scaleList = np.linspace(scaleR, 1, nbScale // 2 + 1).tolist() + np.linspace(1, 1 / scaleR, nbScale // 2 + 1).tolist()[1 :] 
        print (self.scaleList)
        
        torch.cuda.empty_cache()
        
    def ResizeMinSize(self, I, minSize) : 

        w, h = I.size
        ratio = min(w / float(minSize), h / float(minSize)) 
        new_w, new_h = int(round(w/ ratio)), int(round(h / ratio)) 
        new_w, new_h = new_w // self.strideNet * self.strideNet , new_h // self.strideNet * self.strideNet
        
        ratioW, ratioH = new_w / float(w), new_h / float(h)
        Iresize = I.resize((new_w, new_h), resample=Image.LANCZOS)
        
        return Iresize
    
    def setPair(self, Is_org, It_org) : 
        
        with torch.no_grad() : 
            IsList = []
            for i in range(len(self.scaleList)) : 
                IsList.append(self.ResizeMinSize(Is_org, int(self.minSize * self.scaleList[i]) ))
            
            self.Is = IsList[len(self.scaleList) // 2]
            self.IsTensor = self.toTensor(self.Is).unsqueeze(0).cuda()
            
            featsMultiScale = []
            WMultiScale = []
            HMultiScale = []
            for i in range(len(self.scaleList)) : 
                feat = F.normalize(self.net(self.preproc(IsList[i]).unsqueeze(0).cuda()))
                Ws, Hs = outil.getWHTensor(feat)
                featsMultiScale.append( feat.contiguous().view(1024, -1) )
                WMultiScale.append(Ws)
                HMultiScale.append(Hs)
                torch.cuda.empty_cache()
                
            
            
            
            featsMultiScale = torch.cat(featsMultiScale, dim=1)
            WMultiScale = torch.cat(WMultiScale)
            HMultiScale = torch.cat(HMultiScale)
            torch.cuda.empty_cache()
            self.It = self.ResizeMinSize(It_org, self.minSize)
            self.ItTensor = self.toTensor(self.It).unsqueeze(0).cuda()
            featt = F.normalize(self.net(self.preproc(self.It).unsqueeze(0).cuda()))
            Wt, Ht = outil.getWHTensor(featt)
            WtInt, HtInt = outil.getWHTensor_Int(featt)
            
            self.W2, self.H2 = featt.size()[2], featt.size()[3]
            featt = featt.contiguous().view(1024, -1)
            
            index1, index2 = outil.mutualMatching(featsMultiScale, featt)
            self.W1MutualMatch = WMultiScale[index1]
            self.H1MutualMatch = HMultiScale[index1]

            self.W2MutualMatch = Wt[index2]
            self.H2MutualMatch = Ht[index2] 
            
            self.W2MutualMatchInt = WtInt[index2]
            self.H2MutualMatchInt = HtInt[index2] 
            
    
            
            
    def skyFromSeg(self, path) : 
        return self.segNet.getSky(path)
                
    def getCoarse(self, Mt) : 
        ## input mask should be array, 2 dimension, h, w
        with torch.no_grad() :  
            MtExtend = (1 - Mt).astype(np.float32) # 1 is sky, 0 is bulding
            MtExtend = torch.from_numpy(MtExtend).cuda().unsqueeze(0).unsqueeze(0)
            
            MtTensor = F.interpolate(input = MtExtend, size = (self.W2, self.H2), mode = 'bilinear')
            MtTensor = (MtTensor > 0.5).squeeze()
            validMutualMatch = MtTensor[self.W2MutualMatchInt, self.H2MutualMatchInt]
            
            ## RANSAC 
            ones = torch.cuda.FloatTensor(self.W1MutualMatch[validMutualMatch].size(0)).fill_(1)
            match1 = torch.cat((self.H1MutualMatch[validMutualMatch].unsqueeze(1), self.W1MutualMatch[validMutualMatch].unsqueeze(1), ones.unsqueeze(1)), dim=1)
            match2 = torch.cat((self.H2MutualMatch[validMutualMatch].unsqueeze(1), self.W2MutualMatch[validMutualMatch].unsqueeze(1), ones.unsqueeze(1)), dim=1)
            #match1, match2 = match1.cpu().numpy().astype(float), match2.cpu().numpy().astype(float)
            
            if len(match1) < self.nbPoint : 
                return None
            bestParam, _, indexInlier, _ = outil.RANSAC(self.nbIter, match1, match2, self.tolerance, self.nbPoint, self.Transform) 
            if bestParam is None : 
            
                return None
            else : 
            
                return bestParam.astype(np.float32)
