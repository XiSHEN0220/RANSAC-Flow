from coarseAlignFeatMatch import CoarseAlign
import sys 
sys.path.append('../../model')

sys.path.append('../../utils')
import outil

import model as model
import argparse 
import utils
import pickle 
import torch 
import os
import numpy as np 
import pandas as pd
import PIL.Image as Image 

import kornia.geometry as tgm
from scipy.misc import imresize
import torch.nn.functional as F  
from tqdm import tqdm
  
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
    
    
    #match = match12 * F.grid_sample(match21, flowUp)
    match = match12 #* F.grid_sample(match21, flowUp)
    
    match = match * (((flow12.narrow(3, 0, 1) >= -1) * ( flow12.narrow(3, 0, 1) <= 1)).type(torch.cuda.FloatTensor) * ((flow12.narrow(3, 1, 1) >= -1) * ( flow12.narrow(3, 1, 1) <= 1)).type(torch.cuda.FloatTensor)).permute(0, 3, 1, 2) 
    
    match = match[0, 0].cpu().numpy() 
    
    return flow12, match, flowDown8.cpu().numpy(), torch.cat((match12Down8, match21Down8), dim = 1).cpu().numpy()
    
    
# Argument parsing
parser = argparse.ArgumentParser(description='DGC-Net')

parser.add_argument('--csv-path', type=str, default='../../data/Hpatch/csv',
                    help='path to training transformation csv folder')
                    
parser.add_argument('--image-data-path', type=str,
                    default='../../data/Hpatch/hpatches-sequences-release',
                    help='path to folder containing training images')

parser.add_argument('--kernelSize', type=int, default=7, help='Kernel Size')

parser.add_argument('--coarseIter', type=int, default = 50000, help='nb iteration in RANSAC')
parser.add_argument('--maskRegionTh', type=float, default = 0.01, help='if mask region smaller than this value, stop doing homography')
parser.add_argument('--maxCoarse', type=int, default = 10, help='maximum number of coarse alignment')

parser.add_argument('--fineIter', type=int, default = 1000, help='nb iteration in RANSAC')
parser.add_argument('--coarsetolerance', type=float, default = 0.05, help='tolerance coarse in RANSAC')
parser.add_argument('--finetolerance', type=float, default = 0.05, help='tolerance fine in RANSAC')
parser.add_argument('--nbScale', type=int, default=7, choices=[3, 5, 7], help='nb scales ')
parser.add_argument('--minSize', type=int, default= 480, help='min size in the image')

parser.add_argument('--segNet', action='store_true', help='whether to use seg net to remove the sky?')
parser.add_argument('--imageNet', action='store_true', help='whether to use seg net imagenet feature?')

parser.add_argument('--scaleR', type=float, default=2, help='scale range ')

parser.add_argument('--iterR', action='store_true', help='iterative refinement?')


parser.add_argument('--resumePth', type=str, default= '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth', help='Resume Pth file')

parser.add_argument('--transformation', type=str, default= 'Homography', choices=['Affine', 'Homography'], help='transformation')

parser.add_argument('--outDir', type=str, help='output for Global Transformation after refine')



args = parser.parse_args()

print (args)

# Model
# Define Networks
network = {'netFeatCoarse' : model.FeatureExtractor(), 
           'netCorr'       : model.CorrNeigh(args.kernelSize),
           'netFlowCoarse' : model.NetFlowCoarse(args.kernelSize), 
           'netMatch'      : model.NetMatchability(args.kernelSize),
           }
    

for key in list(network.keys()) : 
    network[key].cuda()
    typeData = torch.cuda.FloatTensor

# Network initialization
if args.resumePth:
    param = torch.load(args.resumePth)
    msg = 'Loading pretrained model from {}'.format(args.resumePth)
    print (msg)
    
    for key in list(param.keys()) : 
        network[key].load_state_dict( param[key] ) 
        network[key].eval()

if args.transformation == 'Affine' :
    Transform = outil.Affine
    nbPoint = 3
    
else : 
    Transform = outil.Homography
    nbPoint = 4

outCoarse = args.outDir + '_Coarse'
outFine = args.outDir + '_Fine'

if not os.path.exists(outCoarse) : 
    os.mkdir(outCoarse)
    
if not os.path.exists(outFine) : 
    os.mkdir(outFine)
    
coarseModel = CoarseAlign(args.nbScale, args.coarseIter, args.coarsetolerance, args.transformation, args.minSize, 2, False, args.scaleR, args.imageNet, args.segNet)

   
    
           
with torch.no_grad():
    number_of_scenes = 5
    # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
    for id, k in enumerate(range(2, number_of_scenes + 2)):
        
        outSceneFine = os.path.join( outFine, str(k) )
        outSceneCoarse = os.path.join( outCoarse, str(k) )
            
        if not os.path.exists(outSceneFine) : 
            os.mkdir(outSceneFine)
            
            
        if not os.path.exists(outSceneCoarse) : 
            os.mkdir(outSceneCoarse)
        
        
        csv_file=os.path.join(args.csv_path,'hpatches_1_{}.csv'.format(k))
        
        df = pd.read_csv(csv_file)
        for idx in tqdm(range(len(df))) :
            
            data = df.iloc[idx]
            obj = str(data.obj)
            im1_id, im2_id = str(data.im1), str(data.im2)
            Is = Image.open(os.path.join(args.image_data_path, obj, im1_id + '.ppm')).convert('RGB')
            It = Image.open(os.path.join(args.image_data_path, obj, im2_id + '.ppm')).convert('RGB')
            
            coarseModel.setPair(Is, It)
            
            
            Itw, Ith = coarseModel.It.size
            
            if args.segNet : 
                ## extract bg from segnet 
                It_bg = coarseModel.skyFromSeg( os.path.join(args.image_data_path, obj, im2_id + '.ppm') )
                It_bg = (imresize(It_bg, (Ith, Itw))  < 128).astype(np.float32) ## 0 is bg
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
            
            CoarsePlus_Flow_Tensor = []
            CoarsePlus_Mask_Tensor = []
            
            Fine_Flow_Tensor = []
            Fine_Mask_Tensor = []
            
            FinePlus_Flow_Tensor = []
            FinePlus_Mask_Tensor = []
            nbCoarse = 0
            
            
 
 
            while nbCoarse <= args.maxCoarse : 
                fgMask = ((Mask + (1 - It_bg)) > 0.5).astype(np.float32) ## need to be new region (unmasked, 0 in mask) + fg region (1 in It_bg)
                bestPara = coarseModel.getCoarse(fgMask)
                
                if bestPara is None : 
                    break
                bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
                flowCoarse = warper.warp_grid(bestPara)
                
                flowFine, matchFine, flowFineDown8, matchFineDown8 = PredFlowMask(coarseModel.IsTensor, featt, flowCoarse, grid, network)
            
                
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
                
                np.save(os.path.join(outSceneFine, 'maskBG_' + str(idx) + '_{:d}H.npy'.format(nbCoarse)), It_bg.astype(bool))
                
                
                np.save(os.path.join(outSceneFine, 'mask_' + str(idx) + '_{:d}H.npy'.format(nbCoarse)), Fine_Mask_Tensor)
                
                np.save(os.path.join(outSceneCoarse, 'flow_' + str(idx) + '_{:d}H.npy'.format(nbCoarse)), Coarse_Flow_Tensor)
                np.save(os.path.join(outSceneFine, 'flow_' + str(idx) + '_{:d}H.npy'.format(nbCoarse)), Fine_Flow_Tensor)
                
            
            

            
            
            
        
            
        

        
        
