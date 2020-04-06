import numpy as np
import torch
import kornia.geometry as tgm
import pickle
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import cv2
from tqdm import tqdm
import argparse 
import kornia.geometry as tgm
import pandas as pd

    
def getFlow_all(pairID, finePath, coarsePath, flowList, multiH, warper, grid, th, outW, outH) :
    find = False 
    for flowName in flowList : 
        if flowName.split('_')[1] == str(pairID) : 
            nbH = flowName.split('_')[2].split('H')[0]
            find = True
            break
            
    if not find : 
        return []
        
    flow = torch.from_numpy ( np.load(os.path.join(finePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    param = torch.from_numpy ( np.load(os.path.join(coarsePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    coarse = warper.warp_grid(param)
    
    flow = F.interpolate(input = flow, size = (outH, outW), mode='bilinear')
    flow = flow.permute(0, 2, 3, 1)
    
    flowUp = torch.clamp(flow + grid, min=-1, max=1)
    
    flow = F.grid_sample(coarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()
    
    match = np.load(os.path.join(finePath, 'mask_{:d}_{}H.npy'.format(pairID, nbH)))
    
    
    match = torch.from_numpy(match)
    match = F.interpolate(input = match, size = (outH, outW), mode='bilinear')
    
    #match = match.narrow(1, 0, 1) * F.grid_sample(match.narrow(1, 1, 1), flowUp) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2) 
    
    match = match.narrow(1, 0, 1) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2) 
    
    match = match.permute(0, 2, 3, 1)
        
    flow = torch.clamp(flow, min=-1, max=1)  
    flowGlobal = flow[:1]
      
    if multiH : 
        
        match_binary = match[:1] >= th
    
        for i in range(1, len(match)) : 
            tmp_match = (match.narrow(0, i, 1) >= th) * (~ match_binary)
            match_binary = match_binary + tmp_match 
            tmp_match = tmp_match.expand_as(flowGlobal)
            flowGlobal[tmp_match] = flow.narrow(0, i, 1)[tmp_match]
    
    return flowGlobal
    
def getFlow_onlyCoarse(pairID, finePath, coarsePath, flowList, multiH, warper, grid, th, outW, outH) :
    find = False 
    for flowName in flowList : 
        if flowName.split('_')[1] == str(pairID) : 
            nbH = flowName.split('_')[2].split('H')[0]
            find = True
            break
            
    if not find : 
        return []
        
    flow = torch.from_numpy ( np.load(os.path.join(finePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    param = torch.from_numpy ( np.load(os.path.join(coarsePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    coarse = warper.warp_grid(param)
    
    
    return coarse.narrow(0, 0, 1)
        
def getGT(df, idx, minSize, image_path_orig) : 

    ''' This function is based on the implementation in DGC-Net 
        https://github.com/AaltoVision/DGC-Net
    '''
    data = df.iloc[idx]
    obj = str(data.obj)
    im1_id, im2_id = str(data.im1), str(data.im2)
    h_scale, w_scale = minSize, minSize

    h_ref_orig, w_ref_orig = data.Him.astype('int'), data.Wim.astype('int')
    h_trg_orig, w_trg_orig, _ = \
        cv2.imread(os.path.join(image_path_orig,
                            obj,
                            im2_id + '.ppm'), -1).shape

    H = data[5:].astype('double').values.reshape((3, 3))

    '''
    As gt homography is calculated for (h_orig, w_orig) images,
    we need to
    map it to (h_scale, w_scale)
    H_scale = S * H * inv(S)
    '''
    S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                   [0, h_scale / h_ref_orig, 0],
                   [0, 0, 1]])
    S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                   [0, h_scale / h_trg_orig, 0],
                   [0, 0, 1]])

    H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))

    # inverse homography matrix
    Hinv = np.linalg.inv(H_scale)

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    X, Y = X.flatten(), Y.flatten()

    # create matrix representation
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(Hinv, XYhom)

    # vector representation
    XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
    YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
    ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

    Xwarp = \
        (2 * XwarpHom / (ZwarpHom + 1e-8) / (w_scale - 1) - 1)
    Ywarp = \
        (2 * YwarpHom / (ZwarpHom + 1e-8) / (h_scale - 1) - 1)
    # and now the grid
    grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                           Ywarp.view(h_scale, w_scale)], dim=-1)

    
    return grid_gt.unsqueeze(0)


def epe(input_flow, target_flow):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxWx2]
        target_flow: ground-truth flow [BxHxWx2]
    Output:
        Averaged end-point-error (value)
    """
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()
    
parser = argparse.ArgumentParser()


## model parameters

parser.add_argument('--coarsePth', type=str, help='prediction file ')

parser.add_argument('--finePth', type=str, help='prediction file ')

parser.add_argument('--multiH', action='store_true', help='multiple homograhy or not')

parser.add_argument('--th', type=float, default=1.0, help='threshold for matchability tensor')

parser.add_argument('--minSize', type=int, default= 240, help='min size in the image')

parser.add_argument('--csv-path', type=str, default='../../data/Hpatch/csv',
                    help='path to training transformation csv folder')
                    
parser.add_argument('--image-data-path', type=str,
                    default='../../data/Hpatch/hpatches-sequences-release',
                    help='path to folder containing training images')

parser.add_argument('--onlyCoarse', action='store_true', help='only coarse?')


args = parser.parse_args()
print (args)


strideNet = 16

res = {}

gridY = torch.linspace(-1, 1, steps = args.minSize).view(1, -1, 1, 1).expand(1, args.minSize,  args.minSize, 1)
gridX = torch.linspace(-1, 1, steps = args.minSize).view(1, 1, -1, 1).expand(1, args.minSize,  args.minSize, 1)
grid = torch.cat((gridX, gridY), dim=3)
warper = tgm.HomographyWarper(args.minSize,  args.minSize)
     
test_scene = os.listdir(args.finePth)

getFlow = getFlow_onlyCoarse if args.onlyCoarse else getFlow_all


for scene in test_scene :

    finePath = os.path.join(args.finePth, scene)
    coarsePath = os.path.join(args.coarsePth, scene)
    flowList = os.listdir(finePath)

    print('evaluating for scene {} ....'.format(scene))
    res[scene] = []
    
    csv_file=os.path.join(args.csv_path,'hpatches_1_{}.csv'.format(scene))
        
    df = pd.read_csv(csv_file)
    
    for idx in tqdm(range(len(df))) :
        ## get flow
        
        flow_est = getFlow(idx, finePath, coarsePath, flowList, args.multiH, warper, grid, args.th, args.minSize, args.minSize)
        flow_est = flow_est if len(flow_est) > 0 else grid
        
        ## get gt 
        flow_target = getGT(df, idx, args.minSize, args.image_data_path) 
        
        ## putting flow in the same format
        mask_x_gt = \
            flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = \
            flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3),
                             mask_xx_gt.unsqueeze(3)), dim=3)

        flow_target = (flow_target + 1) * (args.minSize - 1) / (1 + 1)
        flow_est = (flow_est + 1) * (args.minSize - 1) / (1 + 1)

        flow_target_x = flow_target[:, :, :, 0]
        flow_target_y = flow_target[:, :, :, 1]
        flow_est_x = flow_est[:, :, :, 0]
        flow_est_y = flow_est[:, :, :, 1]

        flow_target = \
            torch.cat((flow_target_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                       flow_target_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)
        flow_est = \
            torch.cat((flow_est_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                       flow_est_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)

        ## let's calculate EPE
        aepe = epe(flow_est, flow_target)
        
        res[scene].append(aepe.item())
        
for scene in res : 
    print ('Scene {}, Average end-point error (EPE) : {:.3f}'.format(scene, np.mean(res[scene])))
    



        


