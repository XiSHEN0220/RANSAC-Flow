from coarseAlignFeatMatch import CoarseAlign
import sys 

sys.path.append('../../utils')
import outil

sys.path.append('../../model')
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
from skimage import measure
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def get_info(I) : 
    w, h =I.size
    gridY = torch.linspace(-1, 1, steps = h).view(1, -1, 1, 1).expand(1, h,  w, 1)
    gridX = torch.linspace(-1, 1, steps = w).view(1, 1, -1, 1).expand(1, h,  w, 1)
    grid = torch.cat((gridX, gridY), dim=3).cuda() 
    tensor = transforms.ToTensor()(I).unsqueeze(0).cuda()
    warper = tgm.HomographyWarper(h,  w)
    return w, h, tensor, grid, warper        

def save_tensor(tensor, out_dir, pair_id, img_id) : 
    I = transforms.ToPILImage()(tensor.squeeze().cpu())
    I.save(os.path.join(out_dir, '{}_{}.jpg'.format(pair_id, img_id)))

def save_pil(I, out_dir, pair_id, img_id) : 
    I.save(os.path.join(out_dir, '{}_{}.jpg'.format(pair_id, img_id)))

def save_output(tensor, out_dir, name, pair_id, nbH, data_type) :     
    arr = torch.cat(tensor, dim=0).cpu().numpy().astype(data_type)
    np.save(os.path.join(out_dir, '{}_{}_{}.npy'.format(name, pair_id, nbH)), arr)    
    
def PredFlowMask(IsSample, ItSample, flowCoarse, grid, network) : 

    featsSample = F.normalize(network['netFeatCoarse'](IsSample))
    featt = F.normalize(network['netFeatCoarse'](ItSample))
    
    
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
    
    return flow12, match, flowDown8, torch.cat((match12Down8, match21Down8), dim = 1)



def remove_small_cc(matchFine_finetune, match_th, cc_th):

    if cc_th == 0 :
        return matchFine_finetune
        
    matchFine_Binary = matchFine_finetune > match_th
    all_labels = measure.label(matchFine_Binary, background=0)
    
    if len(np.unique(all_labels)) == 1 : 
        return matchFine_finetune
        
    for i in np.unique(all_labels)[1:] : 
        if np.mean(all_labels == i) <= cc_th :
            matchFine_finetune[all_labels == i] = 0
            
    return matchFine_finetune

## not use this function    
def keep_big_cc(matchFine_finetune, match_th, cc_th):

    if cc_th == 0 :
        return matchFine_finetune
        
    matchFine_Binary = matchFine_finetune > match_th
    all_labels = measure.label(matchFine_Binary, background=0)
    
    if len(np.unique(all_labels)) == 1 : 
        return matchFine_finetune
    
    largest_cc_i, largest_cc_count = 0, 0
    
    for i in np.unique(all_labels)[1:] : 
        cc_count = np.mean(all_labels == i)
        if cc_count > largest_cc_count : 
             largest_cc_count = cc_count
             largest_cc_i = i
    
    match = np.zeros(matchFine_finetune.shape, dtype=np.float32)
    if largest_cc_i > cc_th : 
        match[all_labels == largest_cc_i] = matchFine_finetune[all_labels == largest_cc_i]        
    return match
        
parser = argparse.ArgumentParser()


## model parameters
parser.add_argument('--kernelSize', type=int, default = 7, help='kernel Size')
parser.add_argument('--resumePth', type=str, default = '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth', help='Resume directory')
#parser.add_argument('--resumePth', type=str, default = 'Finetune_Kitti_Train_Model/checkPoint_Epoch4999_Lr0.140_Lf0.00163_Lm0.12295_Lg0.00583', help='Resume directory')

## coarse Align parameters 
parser.add_argument('--coarseIter', type=int, default = 50000, help='nb iteration in RANSAC')
parser.add_argument('--fineIter', type=int, default = 50000, help='nb iteration in the fine alignment RANSAC')

parser.add_argument('--maskRegionTh', type=float, default = 0.005, help='if mask region smaller than this value, stop doing homography')
#parser.add_argument('--maxCoarse', type=int, default = 20, help='maximum number of coarse alignment')
parser.add_argument('--coarsetolerance', type=float, default = 0.05, help='tolerance coarse in RANSAC')
parser.add_argument('--finetolerance', type=float, default = 0.025, help='tolerance coarse in RANSAC')

parser.add_argument('--nbScale', type=int, default=3, choices=[1, 3, 5, 7], help='nb scales ')
parser.add_argument('--scaleR', type=float, default=1.2, help='scale range ')

parser.add_argument('--coarseSize', type=int, default= 800, help='min size in the image')

parser.add_argument('--fineSize', type=int, default= 650, help='min size in the image for fine alignment')

parser.add_argument('--cc_th', type=float, default= 0.01, help='minimum cc size')


## output 
parser.add_argument('--outDir', type=str, help='output directory')

    
parser.add_argument('--segNet', action='store_true', help='whether to use seg net to remove the sky?')
parser.add_argument('--imageNet', action='store_true', help='whether to use seg net imagenet feature?')


## othersize
parser.add_argument('--beginIndex', type=int, default=0, help='begin index')
parser.add_argument('--endIndex', type=int, default=200, help='end index')

subparsers = parser.add_subparsers(title="test dataset", dest="subcommand")

Kitti = subparsers.add_parser("Kitti", help="parser for training arguments")
## test file
Kitti.add_argument('--testImg', type=str, default = '../../data/Kitti/training/image_2/', help='RGB image directory')



args = parser.parse_args()
print (args)



strideNet = 8
Transform = outil.Homography
nbPoint = 4
torch.manual_seed(1000)
np.random.seed(1000)    

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


if not os.path.exists(args.outDir) : 
    os.mkdir(args.outDir)

        
coarseModel = CoarseAlign(args.nbScale, args.coarseIter, args.coarsetolerance, 'Homography', args.coarseSize, 2, False, args.scaleR, args.imageNet, args.segNet)

    
                
## Loading data
nbImg = len(os.listdir(args.testImg)) // 2
indexRoll = torch.cuda.LongTensor([1, 0])
for i in tqdm(range(args.beginIndex, args.endIndex)) :

    Is = Image.open( os.path.join(args.testImg, '{0:06}_11.png'.format(i)) ).convert('RGB')
    It = Image.open( os.path.join(args.testImg, '{0:06}_10.png'.format(i)) ).convert('RGB')
    
    Itw_org, Ith_org = It.size
    
    ## image after resize and its downsampled version
    It_resize = outil.resizeImg(It, strideNet, args.fineSize)
    It_d2 = outil.resizeImg(It, strideNet, args.fineSize // 2)
    
    
    
    with torch.no_grad() : 
    
        w_org, h_org, tensor_org, grid_org, warper_org = get_info(It)
        _, _, tensor_s, _, _ = get_info(Is)
        w_resize, h_resize, tensor_resize, grid_resize, warper_resize = get_info(It_resize)
        w_d2, h_d2, tensor_d2, grid_d2, warper_d2 = get_info(It_d2)
        
    
    
        coarseModel.setPair(Is, It)
        
        ## extract bg from segnet 
        if args.segNet : 
            It_bg = coarseModel.skyFromSeg( os.path.join(args.testImg, '{0:06}_10.png'.format(i)) ) 
            It_bg_tensor = torch.from_numpy((imresize(It_bg, (h_d2, w_d2))  < 128).astype(np.float32)).cuda() ## 0 is bg
            It_bg = (imresize(It_bg, (h_org, w_org))  < 128).astype(np.float32) ## 0 is bg
        else : 
            It_bg = np.ones((h_org, w_org), dtype=np.float32)
            #It_bg_tensor = torch.from_numpy(np.ones((h_d2, w_d2), dtype=np.float32)).cuda() ## 0 is bg
            
        
        
        
        
        ## update mask in every iteration
        Mask = np.zeros((h_org, w_org), dtype=np.float32) # 0 means new region need to be explored, 1 means masked regions
        
        Homography = []
        Org_D2 = []
        Finetune_D2 = []
        Org_Mask = []
        Finetune_Mask = []
        Org = []
        Finetune = []
        
        nbCoarse = 0
    
    while True: #nbCoarse <= args.maxCoarse : 
        fgMask = ((Mask + (1 - It_bg)) > 0.5).astype(np.float32) ## need to be new region (unmasked, 0 in mask) + fg region (1 in It_bg)
        with torch.no_grad() : 
            bestPara = coarseModel.getCoarse(fgMask)
        
        if bestPara is None : 
            break
            
            
        with torch.no_grad() : 
            bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
            homography_d2 = warper_d2.warp_grid(bestPara)
            homography_resize = warper_resize.warp_grid(bestPara)
            
            IsSample_d2 = F.grid_sample(tensor_s, homography_d2)
            
            
        
        with torch.no_grad() : 
            # computing flow for downsampled image
            flowFine, matchFine, flowFine_d2, matchFine_d2 = PredFlowMask(IsSample_d2, tensor_d2, homography_d2, grid_d2, network) # flowFine_d2 need to save, homography need to save as well
            ### 
            
            
            flowCoarse = F.interpolate(flowFine_d2, size=(grid_resize.size()[1], grid_resize.size()[2]), mode='bilinear')
            flowCoarse = flowCoarse.permute(0, 2, 3, 1)
            flowCoarse = torch.clamp(flowCoarse + grid_resize, min=-1, max=1)
            flowCoarse = F.grid_sample(homography_resize.permute(0, 3, 1, 2), flowCoarse).permute(0, 2, 3, 1).contiguous()
            
            IsSample = F.grid_sample(tensor_s, flowCoarse)
            
            # upsampling image
            flowFine_org, matchFine_org, flowFineDown8_org, matchFineDown8_org = PredFlowMask(IsSample, tensor_resize, flowCoarse, grid_org, network) # flowFineDown8_org and matchFineDown8_org need to be saved
            
            #mask = torch.prod(((homography_d2 > -1) * (homography_d2 < 1)).type(torch.cuda.FloatTensor), dim=3) * It_bg_tensor
            #mask = mask.unsqueeze(1)
            
            
            
        
        
        
            flowFine_d2_finetune = flowFine_d2
            flowFineDown8_finetune = flowFineDown8_org
            matchFine_finetune = matchFine_org
            matchFineDown8_finetune = matchFineDown8_org
            
        
        
        # if new region have surface larger than 0.1, save it, otherwise break
        
        matchFine_finetune = remove_small_cc(matchFine_finetune, 0.99, args.cc_th)
        if ((matchFine_finetune > 0.9999) * (1 - fgMask)).mean() > args.maskRegionTh or  nbCoarse == 0:
            ## save all results
            Homography.append(bestPara)
            Finetune_D2.append(flowFine_d2_finetune)
            Finetune_Mask.append(matchFineDown8_finetune)
            Finetune.append(flowFineDown8_finetune) 
            
            
            nbCoarse += 1
            ## update mask 
            matchFine = matchFine_finetune if len(Finetune_Mask) == 0 else matchFine_finetune * (1 - fgMask) 
            Mask = ((Mask + matchFine) > 0.9999).astype(np.float32)
            
        else : 
            break
    if len(Finetune) > 0 :
        nbH = len(Finetune)
        save_output(Homography, args.outDir, 'Homograpy', str(i), nbH, np.float32)
        np.save(os.path.join(args.outDir, 'BG_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), It_bg.astype(bool))
        
        save_output(Finetune_D2, args.outDir, 'Finetune_D2', str(i), nbH, np.float32)
        save_output(Finetune_Mask, args.outDir, 'Finetune_Mask', str(i), nbH, np.float32)
        save_output(Finetune, args.outDir, 'Finetune', str(i), nbH, np.float32)
        
        
            
        
        
        
        
            

