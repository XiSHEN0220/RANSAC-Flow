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
import h5py
from tqdm import tqdm
import argparse 
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
import kornia.geometry as tgm
import json 

def ResizeMinResolution(minSize, I, strideNet): 
    w, h = I.size
    ratio = min(w / float(minSize), h / float(minSize)) 
    new_w, new_h = round(w/ ratio), round(h / ratio) 
    new_w, new_h = new_w // strideNet * strideNet , new_h // strideNet * strideNet
    
    ratioW, ratioH = new_w / float(w), new_h / float(h)
    I = I.resize((new_w, new_h), resample=Image.LANCZOS)
    
    return I

def getResizedSize(minSize, I_size, strideNet):
    w, h = I_size
    ratio = min(w / float(minSize), h / float(minSize)) 
    new_w, new_h = round(w/ ratio), round(h / ratio) 
    new_w, new_h = new_w // strideNet * strideNet , new_h // strideNet * strideNet
        
    return new_w, new_h

## composite image    
def getMeanImage(Is, It): 
    Imean = (np.clip(np.array(Is) / 255. * 0.5 + np.array(It) / 255. * 0.5, a_min = 0.0, a_max = 1.0) * 255 ).astype(np.uint8)
    Imean = Image.fromarray(Imean)
    return Imean

def norm_kp(org_size, new_size, K,  kp):
    """
    Transforms pixel coordinte to image plane coordiante
    """
    w, h = org_size
    w_n, h_n = new_size
    
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    cx += K[0, 2]
    cy += K[1, 2]
    # Get focals
    fx = K[0, 0]
    fy = K[1, 1]

    cx *= (w_n / w)
    cy *= (h_n / h)
    
    fx *= (w_n / w)
    fy *= (h_n / h)

    return (kp - np.array([[cx, cy]])) / np.array([[fx, fy]])

    
        
    
def matches_from_flow(flowFine, matchBinary, sizeA, sizeB, angle):
    
    wA, hA = sizeA
    wB, hB = sizeB
    
    tmp_g_x, tmp_g_y = np.meshgrid(np.arange(wB), np.arange(hB))
    
    gridB = np.stack((tmp_g_x, tmp_g_y), axis=2)
    
    k = angle // 90
    gridB = np.rot90(gridB, k)
    pts2 = gridB[ matchBinary]
    pts1 = flowFine[matchBinary]
    pts1[:, 0] = (pts1[:, 0] + 1) * (wA - 1) / 2
    pts1[:, 1] = (pts1[:, 1] + 1) * (hA - 1) / 2

    return pts1, pts2

    

def opencv_decompose(pts1, pts2, ransac, threshold):
    """
    Estimates and decompose essential matrix with opencv
    """
    res = None
    num_inlier = 0
    mask_final=None
    if pts1.shape[0] >= 5:
        if ransac:
            E, mask_new = cv2.findEssentialMat(
                pts1, pts2, method=cv2.RANSAC, threshold=threshold
            )
        else:
            E, mask_new = cv2.findFundamentalMat(
                pts1, pts2, method=cv2.FM_8POINT
            )

        if E is not None:
            new_RT = False
            # Get the best E just in case we get multiple E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, pts1, pts2, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_final = _mask_new2
                    new_RT = True

            if new_RT:
                res = (R, t)
            else:
                res = None

    return res, mask_final


def evaluate_R_t(R_gt, t_gt, R_pred, t_pred):
    """
    Compares R_pred and t_pred with their ground truth R_gt, t_gt
    """
    t_gt = t_gt.flatten()
    t_pred = t_pred.flatten()

    R = R_gt @ R_pred.T
    err_q = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi

    t_pred = t_pred / (np.linalg.norm(t_pred))
    t_gt = t_gt / (np.linalg.norm(t_gt))

    err_t = np.arccos(t_gt[None, :] @ t_pred[:, None]).item() * 180 / np.pi

    return err_q, err_t
    
    
def getFlow(pairID, finePath, flowList, coarsePath, maskPath, multiH, th) :
    find = False 
    for flowName in flowList :
        if flowName.split('_')[1] == str(pairID) : 
            nbH = flowName.split('_')[2].split('H')[0]
            find = True
            break
            
    if not find : 
        return [], []
        
    flow = torch.from_numpy ( np.load(os.path.join(finePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    param = torch.from_numpy ( np.load(os.path.join(coarsePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))).astype(np.float32) )
    match = np.load(os.path.join(finePath, 'mask_{:d}_{}H.npy'.format(pairID, nbH)))
    matchBG = np.load(os.path.join(maskPath, 'maskBG_{:d}_{}H.npy'.format(pairID, nbH)))
    
    return _getFlow(flow, param, torch.from_numpy(match), matchBG, multiH, th)

def _getFlow(flow, param, match, matchBG, multiH, th):    
    h, w = flow.size()[2], flow.size()[3]
    
    #### -- grid     
    gridY = torch.linspace(-1, 1, steps = h * 8).view(1, -1, 1, 1).expand(1, h * 8,  w * 8, 1)
    gridX = torch.linspace(-1, 1, steps = w * 8).view(1, 1, -1, 1).expand(1, h * 8,  w * 8, 1)
    grid = torch.cat((gridX, gridY), dim=3)
    
    warper = tgm.HomographyWarper(h * 8,  w * 8)
    
    coarse = warper.warp_grid(param)
    
    flow = F.interpolate(input = flow, scale_factor = 8, mode='bilinear')
    flow = flow.permute(0, 2, 3, 1)
    flowUp = torch.clamp(flow + grid, min=-1, max=1)
    
    
    flow = F.grid_sample(coarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()
    
    match = F.interpolate(input = match, scale_factor = 8, mode='bilinear')
    
    match = match.narrow(1, 0, 1) * F.grid_sample(match.narrow(1, 1, 1), flowUp) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2) 
    #match = match.narrow(1, 0, 1) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2)
    
    
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
            
    flowGlobal, match_binary = flowGlobal.squeeze().numpy(), match_binary.squeeze().numpy() * matchBG    
    
    
    return flowGlobal, match_binary

def fix_org_size(org_imsizes_idB, resized_shapes_idB, flow) :
    h, w  = flow.shape[0], flow.shape[1]
    
    w_org, h_org = org_imsizes_idB[0], org_imsizes_idB[1]
    w_resize, h_resize = resized_shapes_idB[0], resized_shapes_idB[1]
    if (h - w) *  (h_org - w_org) >= 0 : 
        return org_imsizes_idB, resized_shapes_idB
    
    else : 
        return (h_org, w_org), (h_resize, w_resize)
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ## model parameters

    parser.add_argument('--gtPath', default = '../../data/YFCC/images/', type=str, help='ground truth file')

    parser.add_argument('--testPair', type=str, default = '../../data/YFCC/pairs', help='RGB image directory')

    parser.add_argument('--multiH', action='store_true', help='multiple homograhy or not')

    parser.add_argument('--ransac', action='store_true', help="using ransac to filter the outlier or not (default False )")

    parser.add_argument('--threshold', type=float, default = 0.0005, help='ransac threshold')

    parser.add_argument('--minSize', type=int, default = 480, help='min size')

    parser.add_argument('--coarsePth', type=str, help='prediction file coarse flow ')

    parser.add_argument('--finePth', type=str, help='prediction file fine flow')

    parser.add_argument('--maskPth', type=str, help='prediction file mask')

    parser.add_argument('--th', type=float, default=0.95, help='threshold for matchability tensor')

    parser.add_argument('--outRes', type=str, default='out.json', help='output json file')

    parser.add_argument('--scene', type=int, choices=[0, 1, 2, 3], help='RGB image directory')

    args = parser.parse_args()
    print ('\n\n\n')
    print (args)
    print ('\n\n\n')


    minSize = args.minSize
    strideNet = 16
    
    scene = ['notre_dame_front_facade', 'buckingham_palace', 'reichstag', 'sacre_coeur']   
    test_scene = [scene[args.scene]]

    res = {}

    for scene in test_scene :

        
        finePath = os.path.join(args.finePth, scene)
        coarsePath = os.path.join(args.coarsePth, scene)
        maskPath = os.path.join(args.maskPth, scene)
        flowList = [item for item in os.listdir(finePath) if 'flow' in item]
        rotation = os.path.join(args.maskPth, scene, 'rotation.json')
        with open(rotation, 'r') as f :
            rotation = json.load(f)
        
        

        print('evaluating for scene {} ....'.format(scene))
        res[scene] = []
        with open(os.path.join(args.testPair, scene + '-te-1000-pairs.pkl'), 'rb') as f :
            pairs_ids = pickle.load(f)


        scene_path = os.path.join(args.gtPath, scene, 'test') 

        with open(os.path.join(scene_path, "images.txt")) as f:
            images_name = [tmp.strip() for tmp in f.readlines()]

        with open(os.path.join(scene_path, "calibration.txt")) as f:
            calib_name = [tmp.strip() for tmp in f.readlines()]

        r_list = list()
        t_list = list()
        geoms = list()
        resized_shapes = list()
        org_imsizes = list()
        K_list = list()

        # Read image infos
        for im, calib in zip(images_name, calib_name):

            calib_h5 = h5py.File(os.path.join(scene_path, calib))      
            r_list.append(np.array(calib_h5["R"]))
            t_list.append(np.array(calib_h5["T"]).T)
            geoms.append(calib_h5)
            org_imsizes.append(np.array(calib_h5['imsize'][0]).tolist())
            
            K_list.append(np.array(calib_h5['K']))


            resized_shapes.append(getResizedSize(minSize, Image.open(os.path.join(scene_path, im)).size, strideNet))
            
            
        #for i, (idA, idB) in tqdm(enumerate(pairs_ids)):
        for i, (idA, idB) in enumerate(pairs_ids):
            
            if i % 50 == 49 : 
                print (i, ' ...')
            
            ## read flow and matchability
            flow, match = getFlow(i, finePath, flowList, coarsePath, maskPath, args.multiH, args.th)
            
            if len(flow) == 0 : 
                res[scene].append(180)
                continue
                
            #org_imsizes[idB], resized_shapes[idB] = fix_org_size(org_imsizes[idB], resized_shapes[idB], flow)
            
            
            # compute relative pose
            r = r_list[idB] @ r_list[idA].T
            t = t_list[idB] - r @ t_list[idA]
            nbH = 11
            
            pts1, pts2 = matches_from_flow(flow, match, resized_shapes[idA], resized_shapes[idB], rotation[str(i)])
            if len(pts1) == 0 :
                res[scene].append(180)
                continue 
            norm_pts1 = norm_kp(org_imsizes[idA], resized_shapes[idA], K_list[idA], pts1)
            norm_pts2 = norm_kp(org_imsizes[idB], resized_shapes[idB], K_list[idB], pts2)

            decomposed, mask = opencv_decompose(norm_pts1, norm_pts2, args.ransac, args.threshold)

            if decomposed is None:
                res[scene].append(180)
            else:
                res[scene].append(max(evaluate_R_t(r, t, decomposed[0], decomposed[1])))
                print (i, max(evaluate_R_t(r, t, decomposed[0], decomposed[1])))
            
                
            
        print ('Scene ', scene, 'Acc@5: ', np.sum(np.array(res[scene]) < 5) / float(len(res[scene])))
        print ('Scene ', scene, 'Acc@10: ', np.sum(np.array(res[scene]) < 10) / float(len(res[scene])))
        print ('Scene ', scene, 'Acc@20: ', np.sum(np.array(res[scene]) < 20) / float(len(res[scene])))
        

    resTotal = []
    for scene in res : 
        resTotal = resTotal + res[scene]
    print ('Total ', 'Acc@5: ', np.sum(np.array(resTotal) < 5) / float(len(resTotal)))
    print ('Total ', 'Acc@10: ', np.sum(np.array(resTotal) < 10) / float(len(resTotal)))
    print ('Total ', 'Acc@20: ', np.sum(np.array(resTotal) < 20) / float(len(resTotal))) 


    with open(args.outRes, 'w') as f : 
        json.dump(res, f)   






