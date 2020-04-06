kernelSize = 7
resumePth = '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth'
minSize = 1600
coarseIter = 10000
maskRegionTh = 0.01
maxCoarse = 10
fineIter = 1000
coarsetolerance = 0.05
finetolerance = 0.05
nbScale = 7
match_th = 0.99
thresholds = {
    'fundamental_ransac': .1,
    "essential_ransac": 1e-3,
    "max_triangle_size": 1e-4
}

use_cuda = False
from pathlib import Path
from PIL import Image
historic_path = Path("/home/fdarmon/various_datasets/history_dataset/notre_dame/")
ps = historic_path / "btv1b53017933x.jpg"
pt = historic_path / "btv1b530179133.jpg"
#ps = Path("/home/fdarmon/OANet/raw_data/yfcc100m/notre_dame_front_facade/test/images/87736245_2352767121.jpg")
Is = Image.open(ps).convert('RGB').crop((200, 100, 900, 1000))
It = Image.open(pt).convert('RGB').crop((200, 100, 900, 1000))

savepath = Path()

import torch
from coarseAlignFeatMatch import CoarseAlign
import sys 
import trimesh
sys.path.append('../../utils')
import outil

sys.path.append('../../model')
import model as model
from evaluation import PredFlowMask, imresize
import os 
import numpy as np
import torch
from torchvision import transforms
import warnings
import torch.nn.functional as F
import pickle 
import sys 
import pandas as pd
import kornia.geometry as tgm
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import trimesh
from getResults import opencv_decompose, _getFlow, matches_from_flow 
from run_point_cloud import compute_and_save

sys.path.append('../')

Transform = outil.Homography
nbPoint = 4


## Loading model
# Define Networks
network = {'netFeatCoarse' : model.FeatureExtractor(),
           'netCorr'       : model.CorrNeigh(kernelSize),
           'netFlowCoarse' : model.NetFlowCoarse(kernelSize),
           'netMatch'      : model.NetMatchability(kernelSize),
           }

if use_cuda:
    device = torch.device("cuda")

    for key in list(network.keys()) :
        network[key].cuda()

else:
    device = torch.device("cpu")
# loading Network
param = torch.load(resumePth)
msg = 'Loading pretrained model from {}'.format(resumePth)
print (msg)

for key in list(param.keys()) :
    network[key].load_state_dict( param[key] )
    network[key].eval()



coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, use_cuda=use_cuda)

# save the cropped source and target image
It.save(savepath / "tmp_It.jpg")
Is.save(savepath / "tmp_Is.jpg")

with torch.no_grad() :
    coarseModel.setSource(Is)
    coarseModel.setTarget(It)

    Itw, Ith = coarseModel.It.size
    Isw, Ish = coarseModel.Is.size

    ## extract bg from segnet
    It_bg_org = coarseModel.skyFromSeg(savepath / "tmp_It.jpg")
    It_bg = 1 - imresize(It_bg_org, (Ith, Itw)).astype(np.float32) ## 0 is bg

if False:
    with torch.no_grad():
        featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))

        #### -- grid
        gridY = torch.linspace(-1, 1, steps = Ith, device=device).view(1, -1, 1, 1).expand(1, Ith,  Itw, 1)
        gridX = torch.linspace(-1, 1, steps = Itw, device=device).view(1, 1, -1, 1).expand(1, Ith,  Itw, 1)
        grid = torch.cat((gridX, gridY), dim=3)
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

        while nbCoarse <= 10 :
            print("Ransac")
            fgMask = ((Mask + (1 - It_bg)) > 0.5).astype(np.float32) ## need to be new region (unmasked, 0 in mask) + fg region (1 in It_bg)
            bestPara, InlierMask = coarseModel.getCoarse(fgMask)

            if bestPara is None :
                break
            bestPara = torch.tensor(bestPara, device=device).unsqueeze(0)
            flowCoarse = warper.warp_grid(bestPara)

            flowFine, matchFine, flowFineDown8, matchFineDown8 = PredFlowMask(coarseModel.IsTensor, featt, flowCoarse, grid, network)
            ### -- Iterative Refine -- ###

            ## Coarse
            matchBinary = (matchFine  * (1 - fgMask)) >= 1.0 # it should be matchable and should be new regions
            ix, iy = np.where(matchBinary)
            gridArr = grid.squeeze().cpu().numpy()
            flow12Arr = flowFine.squeeze().cpu().numpy()

            match2 = np.concatenate((gridArr[ix, iy], np.ones((len(ix), 1))), axis=1)
            match1 = np.concatenate((flow12Arr[ix, iy], np.ones((len(ix), 1))), axis=1)

            if len(match1) < nbPoint :
                break
            coarsePlusParam, _, indexInlier, _ = outil.RANSAC(1000, match1, match2, finetolerance, nbPoint, Transform)

            if coarsePlusParam is None :
                break
            else :

                indexInlier = indexInlier

                ixInlier = ix[indexInlier]
                iyInlier = iy[indexInlier]

                coarsePlusMask = np.zeros((Ith, Itw), dtype=np.float32)
                coarsePlusMask[ixInlier, iyInlier] = 1



            coarsePlusParam = torch.tensor(coarsePlusParam.astype(np.float32), device=device).unsqueeze(0)
            flowCoarsePlus= warper.warp_grid(coarsePlusParam)
            flowFinePlus, matchFinePlus, flowFinePlusDown8, matchFinePlusDown8 = PredFlowMask(coarseModel.IsTensor, featt, flowCoarsePlus, grid, network)
            #matchFinePlus = matchFinePlus * (1 - fgMask)

            if (matchFinePlus * (1 - fgMask)).mean() > maskRegionTh or  nbCoarse == 0:

                ## save coarse
                Coarse_Flow_Tensor.append(bestPara.cpu().numpy())

                ## save fine
                Fine_Flow_Tensor.append(flowFineDown8)
                Fine_Mask_Tensor.append(matchFineDown8)

                ## save coarse Plus
                CoarsePlus_Flow_Tensor.append(coarsePlusParam.cpu().numpy())

                ## save fine Plus
                FinePlus_Flow_Tensor.append(flowFinePlusDown8)
                FinePlus_Mask_Tensor.append(matchFinePlusDown8)

                nbCoarse += 1
                ## update mask
                matchFinePlus = matchFinePlus if len(FinePlus_Mask_Tensor) == 0 else matchFinePlus * (1 - fgMask)
                Mask = ((Mask + matchFinePlus) >= 1.0).astype(np.float32)

            else :
                break

    flow, match, maskBG = _getFlow(
        torch.from_numpy(np.concatenate(Fine_Flow_Tensor, axis=0)),
        torch.from_numpy(np.concatenate(Coarse_Flow_Tensor, axis=0)),
        torch.from_numpy(np.concatenate(Fine_Mask_Tensor, axis=0)),
        matchBG=It_bg.astype(bool),
        multiH=True,
        th=match_th
    )

    compute_and_save(flow, match, Is, It, (Isw, Ish), (Itw, Ith), savepath, thresholds, "ours")

img1 = cv2.imread('tmp_Is.jpg')
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('tmp_It.jpg')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
pts1 = []
pts2 = []

for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=.1)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

H1 = np.empty((3, 3))
H2 = np.empty((3, 3))
cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w2, h2), H1, H2)
dst1 = cv2.warpPerspective(gray1, H1, (w2, h2))
dst2 = cv2.warpPerspective(gray2, H2, (w2, h2))
cv2.imwrite("rectified1.jpg", dst1)
cv2.imwrite("rectified2.jpg", dst2)

input_pt1 = pts1[mask.squeeze().astype(bool)][None]
input_pt2 = pts2[mask.squeeze().astype(bool)][None]

warped1 = cv2.perspectiveTransform(input_pt1, H1).squeeze()
warped2 = cv2.perspectiveTransform(input_pt2, H2).squeeze()

disps = (warped1 - warped2)[:, 0]
min_disp, max_disp = np.percentile(disps, [1, 99])
num_disp = 16 * int((max_disp - min_disp) // 16) + 16

stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
stereo.setMinDisparity(int(min_disp))
stereo.setNumDisparities(num_disp)
disparity = stereo.compute(dst2,dst1)

px_disp = (disparity / 16)

mask_warped2_to_warped1 = px_disp >= int(min_disp)

flow_warped2_to_warped1 = np.ones((h2, w2, 3))
flow_warped2_to_warped1[..., 0] = np.arange(w2)[None, :] - px_disp
flow_warped2_to_warped1[..., 1] = np.arange(h2)[:, None]

flow_warped2_to_org1 = flow_warped2_to_warped1.reshape((-1, 3)) @ np.linalg.inv(H1).T # 1 * 3 * 3 and h * w * 3
flow_warped2_to_org1 = flow_warped2_to_org1[..., :2] / flow_warped2_to_org1[..., 2:]
flow_warped2_to_org1 = flow_warped2_to_org1.reshape((h2, w2, 2))

flow_org2_to_org1 = cv2.warpPerspective(flow_warped2_to_org1, np.linalg.inv(H2), (w2, h2))
flow_org2_to_org1[..., 0] = flow_org2_to_org1[..., 0] * 2 / w2 - 1
flow_org2_to_org1[..., 1] = flow_org2_to_org1[..., 1] * 2 / h2 - 1
mask_org2_to_org1 = cv2.warpPerspective(mask_warped2_to_warped1.astype(float),
                                        np.linalg.inv(H2), (w2, h2)).astype(bool)

maskBG = imresize(It_bg_org, (h2, w2)).astype(bool)

compute_and_save(flow_org2_to_org1,
                 mask_org2_to_org1 * ~maskBG,
                 Is, It, (w1, h1), (w2, h2),
                 savepath, thresholds, "baseline")

