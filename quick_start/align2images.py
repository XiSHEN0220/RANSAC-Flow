# import
from coarseAlignFeatMatch import CoarseAlign
import sys
sys.path.append('../utils/')
import outil
sys.path.append('../model/')
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
import pandas as pd
import kornia.geometry as tgm
from itertools import product
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt 

def get_Avg_Image(Is, It) : 
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))

def align2images(args):
    # Load input images
    img1 = Image.open(args.img1).convert('RGB')
    img2 = Image.open(args.img2).convert('RGB')

    # Load the model
    Transform = outil.Homography
    network = {'netFeatCoarse' : model.FeatureExtractor(), 
               'netCorr'       : model.CorrNeigh(args.kernelSize),
               'netFlowCoarse' : model.NetFlowCoarse(args.kernelSize), 
               'netMatch'      : model.NetMatchability(args.kernelSize),
               }
    for key in list(network.keys()) : 
        network[key].cuda()
        typeData = torch.cuda.FloatTensor

    # Load weights
    param = torch.load(args.resumePth)
    for key in list(param.keys()) : 
        network[key].load_state_dict( param[key] ) 
        network[key].eval()

    # coarse alignment
    coarseModel = CoarseAlign(args.nbScale, args.coarseIter, args.coarsetolerance, 'Homography'
                              , args.minSize, segId = 1, segFg = True, imageNet = True, scaleR = args.scaleR)
    coarseModel.setSource(img1)
    coarseModel.setTarget(img2)
    img2w, img2h = coarseModel.It.size # It: image target.
    # grid
    gridX = torch.linspace(-1, 1, steps = img2w).view(1, 1, -1, 1).expand(1, img2h, img2w, 1)
    gridY = torch.linspace(-1, 1, steps = img2h).view(1, -1, 1, 1).expand(1, img2h, img2w, 1)
    warper = tgm.HomographyWarper(img2h, img2w)
    # compute best parameters
    bestPrm, inlierMask = coarseModel.getCoarse(np.zeros((img2h, img2w)))
    bestPrm = torch.from_numpy(bestPrm).unsqueeze(0).cuda()
    flowCoarse = warper.warp_grid(bestPrm)
    img1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse) #Is: image source.
    img1_coarse_pil = transforms.ToPILImage()(img1_coarse.cpu().squeeze())

    # save for debug
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img1_coarse_pil)
    plt.axis('off')
    plt.title('Source Image (Coarse)')
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Target Image')
    plt.subplot(1, 3, 3)
    plt.imshow(get_Avg_Image(img1_coarse_pil, coarseModel.It))
    plt.axis('off')
    plt.title('Coarse Alignment')
    plt.show()
    plt.savefig(args.outdir + 'comb_coarse_alignment.png')

    # fine alignment
    feat1 = F.normalize(network['netFeatCoarse'](img1_coarse.cuda()))
    feat2 = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
    corr12 = network['netCorr'](feat1, feat2)
    flowDown = network['netFlowCoarse'](corr12, False)
    grid = torch.cat((gridX, gridY), dim = 3).cuda()
    flowUp = F.interpolate(flowDown, size = (grid.size()[1], grid.size()[2]), mode = 'bilinear')
    flowUp = flowUp.permute(0, 2, 3, 1)
    flowUp = flowUp + grid
    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

    img1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
    img1_fine_pil = transforms.ToPILImage()(img1_fine.cpu().squeeze())

    # save for debug
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Fine Alignment)')
    plt.imshow(img1_fine_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(img2)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Overlapped Image')
    plt.imshow(get_Avg_Image(img1_fine_pil, coarseModel.It))
    plt.show()
    plt.savefig(args.outdir + 'comb_fine_alignment.png')

    # save aligned source image
    img1_fine_pil.save(args.outdir + 'fine_aligned_source.png')
    coarseModel.It.save(args.outdir + 'resized_target.png')


if __name__ == '__main__':
    # get arguments with default values.
    parser = argparse.ArgumentParser(description='Align two images')
    parser.add_argument('--img1', type=str, help='path to the first image', default='../img/ArtMiner_Detail_Res13_10.png')
    parser.add_argument('--img2', type=str, help='path to the second image', default='../img/ArtMiner_Detail_Res13_11.png')
    parser.add_argument('--outdir', type=str, help='path to the output folder', default='../output/')
    parser.add_argument('--resumePth', type=str, help='path to the model', default='../model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth')
    parser.add_argument('--kernelSize', type=int, help='size of the kernel', default=7)
    parser.add_argument('--nbPoint', type=int, help='number of points to use for alignment', default=4)
    parser.add_argument('--nbScale', type=int, help='number of scales to use for alignment', default=7)
    parser.add_argument('--coarseIter', type=int, help='number of iterations for coarse alignment', default=10000)
    parser.add_argument('--coarsetolerance', type=float, help='tolerance for coarse alignment', default=0.05)
    parser.add_argument('--minSize', type=int, help='minimum size for the image', default=400)
    parser.add_argument('--scaleR', type=float, help='scale ratio', default=1.2)

    args = parser.parse_args()

    align2images(args)
