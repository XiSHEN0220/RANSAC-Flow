from coarseAlignFeatMatch import CoarseAlign

import sys 
sys.path.append('../../model')

sys.path.append('../../utils')
import outil

import model as model

import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle 
import sys 
import pandas as pd
import kornia.geometry as tgm
from scipy.misc import imresize

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def PredFlowMask(IsTensor, featt, flowCoarse, grid, network) : 

    IsSample = F.grid_sample(IsTensor, flowCoarse)
    featsSample = F.normalize(network['netFeatCoarse'](IsSample))
    
    
    corr12 = network['netCorr'](featt, featsSample)
    flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H
    
    match12Down8 = network['netMatch'](corr12, False)
    
    corr21 = network['netCorr'](featsSample, featt)
    match21Down8 = network['netMatch'](corr21, False)
    
    match12 = F.interpolate(match12Down8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    match21 = F.interpolate(match21Down8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    
    flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    flowUp = flowUp.permute(0, 2, 3, 1)
    
    flowUp = torch.clamp(flowUp + grid, min=-1, max=1)
    
    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()
    
    
    match = match12 * F.grid_sample(match21, flowUp)
    match = match * (((flow12.narrow(3, 0, 1) >= -1) * ( flow12.narrow(3, 0, 1) <= 1)).type(torch.cuda.FloatTensor) * ((flow12.narrow(3, 1, 1) >= -1) * ( flow12.narrow(3, 1, 1) <= 1)).type(torch.cuda.FloatTensor)).permute(0, 3, 1, 2) 
    
    match = match[0, 0].cpu().numpy() 
    
    return flow12, match, flowDown8.cpu().numpy(), torch.cat((match12Down8, match21Down8), dim = 1).cpu().numpy()
    
        
parser = argparse.ArgumentParser()


## model parameters
parser.add_argument('--kernelSize', type=int, default = 7, help='kernel Size')

parser.add_argument('--resumePth', type=str, default = '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth', help='Resume directory')

## Ransac parameters 
parser.add_argument('--minSize', type=int, default = 480, help='minimum size')

parser.add_argument('--transform', type=str, default = 'Homography', choices=['Affine', 'Homography'], help='Homography or Affine transformation')

parser.add_argument('--coarseIter', type=int, default = 10000, help='nb iteration in RANSAC')
parser.add_argument('--maskRegionTh', type=float, default = 0.01, help='if mask region smaller than this value, stop doing homography')
parser.add_argument('--maxCoarse', type=int, default = 10, help='maximum number of coarse alignment')

parser.add_argument('--fineIter', type=int, default = 1000, help='nb iteration in RANSAC')
parser.add_argument('--coarsetolerance', type=float, default = 0.05, help='tolerance coarse in RANSAC')
parser.add_argument('--finetolerance', type=float, default = 0.05, help='tolerance fine in RANSAC')
parser.add_argument('--nbScale', type=int, default=7, choices=[3, 5, 7], help='nb scales ')

parser.add_argument('--outDir', type=str, help='output directory')

parser.add_argument('--segNet', action='store_true', help='whether to use seg net to remove the sky?')
parser.add_argument('--imageNet', action='store_true', help='whether to use seg net imagenet feature?')

parser.add_argument('--scaleR', type=float, default=2, help='scale range ')


subparsers = parser.add_subparsers(title="test dataset", dest="subcommand")

robotCar = subparsers.add_parser("RobotCar", help="parser for training arguments")

## test file
robotCar.add_argument('--testDir', type=str, default = '../../data/RobotCar/imgs/', help='RGB image directory')
robotCar.add_argument('--testCSV', type=str, default = '../../data/RobotCar/test6511.csv', help='RGB image directory')
robotCar.add_argument('--beginIndex', type=int, default = 0, help='begin index')
robotCar.add_argument('--endIndex', type=int, default = 6511, help='end index')


megaDepth1600 = subparsers.add_parser("MegaDepth", help="parser for training arguments")

## test file
megaDepth1600.add_argument('--testDir', type=str, default = '../../data/MegaDepth/Test/test1600Pairs', help='RGB image directory')
megaDepth1600.add_argument('--testCSV', type=str, default = '../../data/MegaDepth/Test/test1600Pairs.csv', help='RGB image directory')
megaDepth1600.add_argument('--beginIndex', type=int, default = 0, help='begin index')
megaDepth1600.add_argument('--endIndex', type=int, default = 1600, help='end index')

args = parser.parse_args()
print (args)


 
std = 5

Transform = outil.Homography
nbPoint = 4
    

## Loading model
# Define Networks
network = {'netFeatCoarse' : model.FeatureExtractor(), 
           'netCorr'       : model.CorrNeigh(args.kernelSize),
           'netFlowCoarse' : model.NetFlowCoarse(args.kernelSize), 
           'netMatch'      : model.NetMatchability(args.kernelSize),
           }
    

for key in list(network.keys()) : 
    network[key].cuda()
    typeData = torch.cuda.FloatTensor

# loading Network 
if args.resumePth:
    param = torch.load(args.resumePth)
    msg = 'Loading pretrained model from {}'.format(args.resumePth)
    print (msg)
    
    for key in list(param.keys()) : 
        network[key].load_state_dict( param[key] ) 
        network[key].eval()


outCoarse = args.outDir + '_Coarse'
outFine = args.outDir + '_Fine'

if not os.path.exists(outCoarse) : 
    os.mkdir(outCoarse)
    
if not os.path.exists(outFine) : 
    os.mkdir(outFine)


        
coarseModel = CoarseAlign(args.nbScale, args.coarseIter, args.coarsetolerance, 'Homography', args.minSize, 2, False, args.scaleR, args.imageNet, args.segNet)



## Loading data    
# Set up for real validation
df = pd.read_csv(args.testCSV, dtype=str)

                                                     
with torch.no_grad() : 
    for i in tqdm(range(args.beginIndex, args.endIndex)) : 
        
        scene = df['scene'][i]
        
        #### --  Source Image feature
        Ispath = os.path.join( os.path.join(args.testDir, scene), df['source_image'][i]) if scene != '/' else os.path.join( args.testDir, df['source_image'][i])
        Itpath = os.path.join( os.path.join(args.testDir, scene), df['target_image'][i]) if scene != '/' else os.path.join( args.testDir, df['target_image'][i])
        
        Is = Image.open( Ispath ).convert('RGB') 
        It = Image.open( Itpath ).convert('RGB') 
        
        #### -- Target Image feature
        
        coarseModel.setPair(Is, It)
        
        Itw, Ith = coarseModel.It.size
        ## extract bg from segnet 
        if args.segNet : 
            It_bg = coarseModel.skyFromSeg( Itpath ) # 1 is bg
        
            It_bg = (imresize(It_bg, (coarseModel.It.size[1], coarseModel.It.size[0]))  < 128).astype(np.float32) ## 0 is bg
        else : 
            It_bg = np.ones((Ith, Itw), dtype=np.float32)
        
        featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
        
        #### -- grid     
        gridY = torch.linspace(-1, 1, steps = Ith).view(1, -1, 1, 1).expand(1, Ith,  Itw, 1)
        gridX = torch.linspace(-1, 1, steps = Itw).view(1, 1, -1, 1).expand(1, Ith,  Itw, 1)
        grid = torch.cat((gridX, gridY), dim=3).cuda() 
        warper = tgm.HomographyWarper(Ith,  Itw)
        
        ## update mask in every iteration
        Mask = np.zeros((Ith, Itw), dtype=np.float32) # 0 means new region need to be explored, 1 means masked regions
        
        Coarse_Flow_Tensor = []
        Coarse_Mask_Tensor = []
        
        
        Fine_Flow_Tensor = []
        Fine_Mask_Tensor = []
        
        nbCoarse = 0
        
        while nbCoarse <= args.maxCoarse : 
            fgMask = ((Mask + (1 - It_bg)) > 0.5).astype(np.float32) ## need to be new region (unmasked, 0 in mask) + fg region (1 in It_bg)
            
            bestPara = coarseModel.getCoarse(fgMask)
            
            if bestPara is None : 
                break
            bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
            flowCoarse = warper.warp_grid(bestPara)
            
            flowFine, matchFine, flowFineDown8, matchFineDown8 = PredFlowMask(coarseModel.IsTensor, featt, flowCoarse, grid, network)
            
            #matchFinePlus = matchFinePlus * (1 - fgMask)
            
            # if new region have surface larger than 0.1, save it, otherwise break
            
            if (matchFine * (1 - fgMask)).mean() > args.maskRegionTh or  nbCoarse == 0:
                
                ## save coarse 
                    Coarse_Flow_Tensor.append(bestPara.cpu().numpy())
                    
                    ## save fine 
                    Fine_Flow_Tensor.append(flowFineDown8)
                    Fine_Mask_Tensor.append(matchFineDown8)
                    
                    
                    nbCoarse += 1
                    ## update mask 
                    matchFine = matchFine if len(Fine_Mask_Tensor) == 0 else matchFine * (1 - fgMask) 
                    Mask = ((Mask + matchFine) >= 1.0).astype(np.float32)
                    
            else : 
                break
                    
        if len(Fine_Mask_Tensor) > 0 :
            

            Fine_Mask_Tensor = np.concatenate(Fine_Mask_Tensor, axis=0)
            
            
            Coarse_Flow_Tensor = np.concatenate(Coarse_Flow_Tensor, axis=0)
            Fine_Flow_Tensor = np.concatenate(Fine_Flow_Tensor, axis=0)
            
            np.save(os.path.join(outFine, 'maskBG_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), It_bg.astype(bool))
            
            
            np.save(os.path.join(outFine, 'mask_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Fine_Mask_Tensor)
            
            np.save(os.path.join(outCoarse, 'flow_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Coarse_Flow_Tensor)
            np.save(os.path.join(outFine, 'flow_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Fine_Flow_Tensor)
            
            
