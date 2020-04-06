
from tqdm import tqdm 
import torch.nn.functional as F
import torch 
import numpy as np 

import kornia.geometry as tgm

import sys 
sys.path.append('../../model')
import model as model

sys.path.append('../../utils')
import outil

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

def iterative(net, gridFine, source_img, target_img, flow_est, i, match_est, Transform, coarsePlus, transformation, nbIter = 1000, tolerance=0.03, nbPoint = 4):

    if i not in coarsePlus : 
        match_est = ( match_est * (((flow_est.narrow(3, 0, 1) >= -1) * ( flow_est.narrow(3, 0, 1) <= 1)).type(torch.cuda.FloatTensor) * ((flow_est.narrow(3, 1, 1) >= -1) * ( flow_est.narrow(3, 1, 1) <= 1)).type(torch.cuda.FloatTensor)).permute(0, 3, 1, 2) ).squeeze().cpu().numpy()
        match_est = (match_est > 0.5)
        
        ix, iy = np.where(match_est)
        gridArr = gridFine.squeeze().cpu().numpy()
        flow_estArr = flow_est.squeeze().cpu().numpy()
        match2 = np.concatenate((gridArr[ix, iy], np.ones((len(ix), 1))), axis=1)
        match1 = np.concatenate((flow_estArr[ix, iy], np.ones((len(ix), 1))), axis=1)
        if len(match1) > nbPoint : 
            bestParam, bestInlier, match1Inlier, match2Inlier = outil.RANSAC(nbIter, match1, match2, tolerance, nbPoint, Transform)
        else : 
            bestParam = np.eye(3)
            
        bestParam = bestParam.astype(np.float32)
        coarsePlus[i] = bestParam 
     
    else : 
        bestParam = coarsePlus[i]
    
    if transformation == 'Affine' : 
        grid = F.affine_grid(torch.from_numpy(bestParam[:2].astype(np.float32)).unsqueeze(0).cuda(), target_img.size()) # theta should be of size N×2×3
    else : 
        bestParam = bestParam.astype(np.float32)  
        warper = tgm.HomographyWarper(target_img.size()[2], target_img.size()[3])
        grid = warper.warp_grid(torch.from_numpy(bestParam).unsqueeze(0).cuda())
        
    IsSample = F.grid_sample(source_img, grid)
    
    sourceFeat = F.normalize(net['netFeatCoarse'](IsSample))
    targetFeat = F.normalize(net['netFeatCoarse'](target_img))

    
    corr12 = net['netCorr'](targetFeat, sourceFeat)
    
    _, flow_est =  model.predFlowCoarse(corr12, net['netFlowCoarse'], gridFine) 
    flow_est = F.grid_sample(grid.permute(0, 3, 1, 2), flow_est).permute(0, 2, 3, 1).contiguous()
    return flow_est
    
        
        
    
    
    

def calculate_epe_hpatches(net, val_loader, device, k, inPklCoarse, onlyCoarse, transformation, Transform, coarsePlus = None, iterativeRefine=False, img_size=240):
    """
    Compute EPE for HPatches dataset
    Args:
        net: trained model
        val_loader: input dataloader
        device: `cpu` or `gpu`
        img_size: size of input images
    Output:
        aepe_array: averaged EPE for the whole sequence of HPatches
    """
    aepe_array = []
    n_registered_pxs = 0
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, mini_batch in pbar:
        
        source_img = mini_batch['source_image'].to(device)
        target_img = mini_batch['target_image'].to(device)
        bs, _, _, _ = source_img.shape

        #### 
        # net prediction
        gridY = torch.linspace(-1, 1, steps = target_img.size(2)).view(1, -1, 1, 1).expand(1, target_img.size(2),  target_img.size(3), 1)
        gridX = torch.linspace(-1, 1, steps = target_img.size(3)).view(1, 1, -1, 1).expand(1, target_img.size(2),  target_img.size(3), 1)
        gridFine = torch.cat((gridX, gridY), dim=3).cuda() 
        
        
        bestParam = inPklCoarse[i]
        if transformation == 'Affine' : 
            grid = F.affine_grid(torch.from_numpy(bestParam[:2].astype(np.float32)).unsqueeze(0).cuda(), target_img.size()) # theta should be of size N×2×3
        else : 
            bestParam = bestParam.astype(np.float32)  
            warper = tgm.HomographyWarper(target_img.size()[2], target_img.size()[3])
            grid = warper.warp_grid(torch.from_numpy(bestParam).unsqueeze(0).cuda())
            
        IsSample = F.grid_sample(source_img, grid)
        
        sourceFeat = F.normalize(net['netFeatCoarse'](IsSample))
        targetFeat = F.normalize(net['netFeatCoarse'](target_img))

        
        corr12 = net['netCorr'](targetFeat, sourceFeat)
        match12 = model.predMatchability(corr12, net['netMatch']) 
        
        corr21 = net['netCorr'](sourceFeat, targetFeat)
        match21 = model.predMatchability(corr21, net['netMatch']) 
        
        
        
        _, flow_est =  model.predFlowCoarse(corr12, net['netFlowCoarse'], gridFine) 
        match_est = match12 * F.grid_sample(match21, flow_est) 
        _, flow_est_inverse =  model.predFlowCoarse(corr21, net['netFlowCoarse'], gridFine) 
        
        flow_est = F.grid_sample(grid.permute(0, 3, 1, 2), flow_est).permute(0, 2, 3, 1).contiguous()
        flow_est = grid if onlyCoarse else flow_est
        
        if iterativeRefine : 
            flow_est = iterative(net, gridFine, source_img, target_img, flow_est, i, match_est, Transform, coarsePlus, transformation)
        
        
         
        flow_target = mini_batch['correspondence_map'].to(device)
        
        mask_x_gt = \
            flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = \
            flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3),
                             mask_xx_gt.unsqueeze(3)), dim=3)

        for i in range(bs):
            # unnormalize the flow: [-1; 1] -> [0; im_size - 1]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / (1 + 1)
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1) / (1 + 1)

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

        # let's calculate EPE
        aepe = epe(flow_est, flow_target)
        aepe_array.append(aepe.item())
        n_registered_pxs += flow_target.shape[0]
        
    return aepe_array
