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

def align2imgs(args):
    # Load input images
    img1 = Image.open(args.img1).convert('RGB')
    img2 = Image.open(args.img2).convert('RGB')
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Target Image')
    plt.title('input imagees')
    plt.show()
    plt.savefig(args.outdir + 'input_images.png')

    # Load the model
    resumePth = args.resumePth
    kernelSize = 7
    Transform = outil.Homography
    nbPoint = 4
    network = {'netFeatCoarse' : model.FeatureExtractor(), 
               'netCorr'       : model.CorrNeigh(kernelSize),
               'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
               'netMatch'      : model.NetMatchability(kernelSize),
               }
    for key in list(network.keys()) : 
        network[key].cuda()
        typeData = torch.cuda.FloatTensor

    pass

# in case called as a script
if __name__ == '__main__':
    # get arguments with default values.
    parser = argparse.ArgumentParser(description='Align two images')
    parser.add_argument('--img1', type=str, help='path to the first image', default='../img/ArtMiner_Detail_Res13_10.png')
    parser.add_argument('--img2', type=str, help='path to the second image', default='../img/ArtMiner_Detail_Res13_11.png')
    parser.add_argument('--outdir', type=str, help='path to the output folder', default='../output/')
    parser.add_argument('--resumePth', type=str, help='path to the model', default='../model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth')
    args = parser.parse_args()

    align2imgs(args)
