import numpy as np
import torch
import kornia.geometry as tgm
import pickle
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse 
import kornia.geometry as tgm
from skimage import measure
from scipy import ndimage as nd
from scipy import misc 
def readFlow(path) : 
    flow = cv2.imread(path, cv2.IMREAD_UNCHANGED) ## Read as BGR, R for U, G for V, B for valid
    valid, V, U = flow[:, :, 0], flow[:, :, 1], flow[:, :, 2]

    U =  (U.astype(float) - 32768) / 64.0
    V =  (V.astype(float) - 32768) / 64.0
    valid = valid.astype(bool)    
    return U, V, valid

def get_imgsize(w, h, strideNet, minSize = 400) :

    ## resize img, the largest dimension is maxSize
    wratio, hratio = w / minSize, h / minSize
    resizeRatio = min(wratio, hratio)
    
    w, h= w / resizeRatio, h / resizeRatio
    
    resizeW = round(w/ strideNet) * strideNet
    resizeH = round(h/ strideNet) * strideNet
    
    
    return resizeW, resizeH

## not use this function
def keep_big_cc(matchFine_finetune, cc_th, match_th = 0.99):
    if cc_th == 0 :
        return matchFine_finetune
    
    matchFine_finetune = matchFine_finetune.squeeze().numpy() 
    match = np.zeros(matchFine_finetune.shape, dtype=np.float32)
    for j in range(matchFine_finetune.shape[0]) : 
        match_j =  matchFine_finetune[j]
        
        matchFine_Binary = match_j > match_th
        
        all_labels = measure.label(matchFine_Binary, background=0)
        
        if len(np.unique(all_labels)) == 1 : 
            break
        largest_cc_j, largest_cc_count = 0, 0
        for i in np.unique(all_labels)[1:] : 
            cc_count = np.mean(all_labels == i)
            if cc_count > largest_cc_count : 
                 largest_cc_count = cc_count
                 largest_cc_j = i
        if largest_cc_j > cc_th : 
            match[j][all_labels == largest_cc_j] = matchFine_finetune[j][all_labels == largest_cc_j]    
    return torch.from_numpy(matchFine_finetune).unsqueeze(1)
        
def remove_small_cc(matchFine_finetune, cc_th, match_th = 0.99):
    if cc_th == 0 :
        return matchFine_finetune
    
    matchFine_finetune = matchFine_finetune.squeeze(1).numpy() 
    for j in range(matchFine_finetune.shape[0]) : 
        match_j =  matchFine_finetune[j]
        matchFine_Binary = match_j > match_th
        all_labels = measure.label(matchFine_Binary, background=0)
        
        if len(np.unique(all_labels)) == 1 : 
            continue
            
        for i in np.unique(all_labels)[1:] : 
            if np.mean(all_labels == i) <= cc_th :
                match_j[all_labels == i] = 0
        matchFine_finetune[j] = match_j
            
    return torch.from_numpy(matchFine_finetune).unsqueeze(1)
    

def interpolate_flow_match(flowGlobal, match_binary) : 
    match_binary = (~match_binary).squeeze().numpy()
    idx = nd.distance_transform_edt(match_binary, return_distances=False, return_indices=True)
    flowGlobal = flowGlobal.squeeze().numpy()
    flowGlobal = flowGlobal[tuple(idx)]
    
    return torch.from_numpy(flowGlobal).unsqueeze(0)
    
def getFlow_all(pairID, predDir, nbH, res_name, warper_org, multiH, grid_org, th, cc_th, interpolate) :
    
        
    param = torch.from_numpy ( np.load(os.path.join(predDir, 'Homograpy_{}_{}.npy'.format(pairID, nbH))).astype(np.float32) )
    flowd2 = torch.from_numpy ( np.load(os.path.join(predDir, '{}_D2_{}_{}.npy'.format(res_name, pairID, nbH))).astype(np.float32) )
    flow = torch.from_numpy ( np.load(os.path.join(predDir, '{}_{}_{}.npy'.format(res_name, pairID, nbH))).astype(np.float32) )
    
    homography_org = warper_org.warp_grid(param)
    flowd2 = F.interpolate(flowd2, size=(grid_org.size()[1], grid_org.size()[2]), mode='bilinear')
    #flowd2 = torch.zeros(flow.size()[0],1,grid_org.size()[1], grid_org.size()[2])
    flowd2 = flowd2.permute(0, 2, 3, 1)
    flowd2 = torch.clamp(flowd2 + grid_org, min=-1, max=1)
    flowd2 = F.grid_sample(homography_org.permute(0, 3, 1, 2), flowd2).permute(0, 2, 3, 1).contiguous()


    flow = F.interpolate(flow, size=(grid_org.size()[1], grid_org.size()[2]), mode='bilinear')
    flow = flow.permute(0, 2, 3, 1)
    flowUp = torch.clamp(flow + grid_org, min=-1, max=1)
    flow = F.grid_sample(flowd2.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()
    
    match = np.load(os.path.join(predDir, '{}_Mask_{}_{}.npy'.format(res_name, pairID, nbH)))
    matchBG = np.load(os.path.join(predDir, 'BG_{}_{}H.npy'.format(pairID, nbH)))
    
    
    
    match = torch.from_numpy(match)
    match = F.interpolate(input = match, size = (grid_org.size()[1], grid_org.size()[2]), mode='bilinear')
    
    match = match.narrow(1, 0, 1) * F.grid_sample(match.narrow(1, 1, 1), flowUp) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2) 
    
    match = remove_small_cc(match, cc_th)
    match = match.permute(0, 2, 3, 1)
    flow = torch.clamp(flow, min=-1, max=1)  
    flowGlobal = flow[:1]
    match_binary = match[:1] >= th
    if multiH : 
        
        
        for i in range(1, len(match)) : 
                
            tmp_match = (match.narrow(0, i, 1) >= th) * (~ match_binary)
            match_binary = match_binary + tmp_match 
            tmp_match = tmp_match.expand_as(flowGlobal)
            flowGlobal[tmp_match] = flow.narrow(0, i, 1)[tmp_match]
    if interpolate : 
        flowGlobal = interpolate_flow_match(flowGlobal, match_binary)
    return flowGlobal
        
        
def getFlow_onlyCoarse(pairID, predDir, nbH, res_name, warper_org, multiH, grid_org, th, cc_th, interpolate) :
    
        
    param = torch.from_numpy ( np.load(os.path.join(predDir, 'Homograpy_{}_{}.npy'.format(pairID, nbH))).astype(np.float32) )
    
    homography_org = warper_org.warp_grid(param)
    
    return homography_org.narrow(0, 0, 1) 
        



parser = argparse.ArgumentParser()


## model parameters

parser.add_argument('--gtPath', default = '../../data/Kitti/training/flow_noc/', type=str, help='ground truth file')

parser.add_argument('--predDir', type=str, help='prediction dir ')

parser.add_argument('--resName', type=str, choices=['Org', 'Finetune'], default='Finetune', help='evaluate which results, Org or Finetune')

parser.add_argument('--multiH', action='store_true', help='multiple homograhy or not')

parser.add_argument('--th', type=float, default=1.0, help='threshold for matchability tensor')

parser.add_argument('--cc_th', type=float, default=0.01, help='threshold of the smallest connected component')

parser.add_argument('--noc', action='store_true', help='non occluded region or not')

parser.add_argument('--interpolate', action='store_true', help='interpolate or not ?')

parser.add_argument('--onlyCoarse', action='store_true', help='only coarse?')

args = parser.parse_args()
print (args)


strideNet = 8

res = []

nbImg = 200

bg = [item for item in os.listdir(args.predDir) if 'BG' in item]
bg = [(item.split('_')[1], item.split('_')[2].split('H')[0]) for item in bg]
dict_pairid_nbH = dict(bg)
print (dict_pairid_nbH)

if args.noc : 
    args.gtPath =  '../../data/Kitti/training/flow_noc/'
else : 
    args.gtPath =  '../../data/Kitti/training/flow_occ/'
    
getFlow = getFlow_onlyCoarse if args.onlyCoarse else getFlow_all

for i in tqdm(range(nbImg)) :
    
    path = os.path.join(args.gtPath, '{0:06}_10.png'.format(i))
    u, v, valid = readFlow(path)
    Ith, Itw = u.shape[0], u.shape[1]
    warper_org = tgm.HomographyWarper(Ith,  Itw)
    
    #### -- org grid
    gridY = torch.linspace(-1, 1, steps = Ith).view(1, -1, 1, 1).expand(1, Ith,  Itw, 1)
    gridX = torch.linspace(-1, 1, steps = Itw).view(1, 1, -1, 1).expand(1, Ith,  Itw, 1)
    grid_org = torch.cat((gridX, gridY), dim=3)
    find = True
    if str(i) not in dict_pairid_nbH : 
        flow = grid_org
        find = False
    else : 
        nbH = dict_pairid_nbH[str(i)]
        flow = getFlow(str(i), args.predDir, nbH, args.resName, warper_org, args.multiH, grid_org, args.th, args.cc_th, args.interpolate)
    
    
    flow = flow - grid_org
    flow = flow.numpy()
    upred = flow[0, :, :, 0] * (Itw - 1) / 2
    vpred = flow[0, :, :, 1] * (Ith - 1) / 2
    
    error = ((upred - u) ** 2 + (vpred - v) ** 2) ** 0.5 
    
    avg_error = np.sum(error * valid) / np.sum(valid)
    
    res.append(avg_error)
    print (i, np.mean(res))
    if not find : 
        raise RuntimeError('XXX')
    
print ('Average end-point error (EPE) : ', np.mean(res))
    



        


