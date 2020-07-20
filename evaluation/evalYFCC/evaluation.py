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
import json 

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#def imresize(im, size):
#    return np.array(Image.fromarray(im).resize((size[1], size[0])))
    
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
    match = match * (((flow12.narrow(3, 0, 1) >= -1) * ( flow12.narrow(3, 0, 1) <= 1)).float() * ((flow12.narrow(3, 1, 1) >= -1) * ( flow12.narrow(3, 1, 1) <= 1)).float()).permute(0, 3, 1, 2)
    
    match = match[0, 0].cpu().numpy() 
    
    return flow12, match, flowDown8.cpu().numpy(), torch.cat((match12Down8, match21Down8), dim = 1).cpu().numpy()
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()


    ## model parameters
    parser.add_argument('--kernelSize', type=int, default = 7, help='kernel Size')

    parser.add_argument('--resumePth', type=str, default = '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth', help='Resume directory')
    #parser.add_argument('--resumePth', type=str, default = '../../model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth', help='Resume directory')

    ## Others 
    parser.add_argument('--minSize', type=int, default = 480, help='minimum size')

    parser.add_argument('--coarseIter', type=int, default = 10000, help='nb iteration in RANSAC')
    parser.add_argument('--maskRegionTh', type=float, default = 0.01, help='if mask region smaller than this value, stop doing homography')
    parser.add_argument('--maxCoarse', type=int, default = 10, help='maximum number of coarse alignment')

    parser.add_argument('--coarsetolerance', type=float, default = 0.05, help='tolerance coarse in RANSAC')
    parser.add_argument('--nbScale', type=int, default=7, choices=[3, 5, 7], help='nb scales ')
    parser.add_argument('--outDir', type=str, help='output directory')

    parser.add_argument('--segNet', action='store_true', help='whether to use seg net to remove the sky?')
    parser.add_argument('--imageNet', action='store_true', help='whether to use seg net imagenet feature?')
    
    parser.add_argument('--scaleR', type=float, default=2, help='scale range ')


    subparsers = parser.add_subparsers(title="test dataset", dest="subcommand")

    YFCC = subparsers.add_parser("YFCC", help="parser for training arguments")

    ## test file
    YFCC.add_argument('--testImg', type=str, default = '../../data/YFCC/images', help='RGB image directory')
    YFCC.add_argument('--testPair', type=str, default = '../../data/YFCC/pairs', help='RGB image directory')
    YFCC.add_argument('--beginIndex', type=int, default = 0, help='begin index')
    YFCC.add_argument('--endIndex', type=int, default = 1000, help='end index')
    YFCC.add_argument('--testScene', type=str, choices=['notre_dame_front_facade', 'buckingham_palace', 'reichstag', 'sacre_coeur'], help='RGB image directory')


    args = parser.parse_args()
    print (args)



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


            
    coarseModel = CoarseAlign(args.nbScale, args.coarseIter, args.coarsetolerance, 'Homography', args.minSize, segId = 1, segFg = True, use_cuda=True, imageNet = args.imageNet, segNet = args.segNet, scaleR = args.scaleR)
    
            
    ## Loading data
    sceneList = os.listdir(args.testPair)
    sceneList = sceneList if not args.testScene else [item for item in sceneList if args.testScene in item] 

    for scene in sceneList :

        sceneName = scene.split('-te')[0]
        print ('Scene {} ...'.format(sceneName)) 
        
        with open(os.path.join(args.testPair, scene), 'rb') as f :
            df = pickle.load(f)
        
        imgDir = os.path.join(args.testImg,  sceneName, 'test')
        imgList = open(os.path.join(args.testImg,  sceneName, 'test', 'images.txt'), 'r').read().split('\n')[:-1]  

        outSceneFine = os.path.join( outFine, sceneName )
        outSceneCoarse = os.path.join( outCoarse, sceneName )
        outRotation =  os.path.join( outFine, sceneName, 'rotation.json')   
        
        if not os.path.exists(outSceneFine) : 
            os.mkdir(outSceneFine)
            
        if not os.path.exists(outSceneCoarse) : 
            os.mkdir(outSceneCoarse)
        
        

        
        angle_list = [0, 90, 180, 270]  
        angle_rotation = {}                                               
        with torch.no_grad() : 
            for i in tqdm(range(args.beginIndex, args.endIndex)) : 
                Is = Image.open( os.path.join(imgDir, imgList[df[i][0]]) ).convert('RGB')
                coarseModel.setSource(Is)
                
                It = Image.open( os.path.join(imgDir, imgList[df[i][1]]) ).convert('RGB')
                
                if args.segNet : 
                    It_bg = coarseModel.skyFromSeg( os.path.join(imgDir, imgList[df[i][1]]) )
                    
                else : 
                    It_bg = np.ones((It.size[1], It.size[0]), dtype=np.float32)
                ## fix rotation pb 
                ItList = [It, It.rotate(90, expand=True), It.rotate(180, expand=True), It.rotate(270, expand=True)]
                
                It_bg_List = [It_bg, np.rot90(It_bg), np.rot90( It_bg, 2 ), np.rot90( It_bg , 3)]
                
                nbInlier = []
                for j in range(4) : 
                    coarseModel.setTarget(ItList[j])
                    Itw, Ith = coarseModel.It.size
                    It_bg = It_bg_List[j]
                    It_bg = (imresize(It_bg, (Ith, Itw))  < 128).astype(np.float32) ## 0 is bg #if args.segNet else np.ones((Ith, Itw), dtype=np.float32)
                    fgMask = (( 1 - It_bg) > 0.5).astype(np.float32)
                    bestPara, InlierMask = coarseModel.getCoarse(fgMask)
                    if bestPara is None : 
                        nbInlier.append(0)
                    else : 
                        nbInlier.append( np.sum(InlierMask) )
                
                coarseModel.setTarget(ItList[np.argmax(nbInlier)])
                angle_rotation[i] = angle_list[np.argmax(nbInlier)]
                It_bg = It_bg_List[np.argmax(nbInlier)]
                Itw, Ith = coarseModel.It.size
                It_bg = (imresize(It_bg, (Ith, Itw))  < 128).astype(np.float32) if args.segNet else np.ones((Ith, Itw), dtype=np.float32)
                #print (It_bg.mean())
                    
                
                ## extract bg from segnet 
                
                
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
                    bestPara, InlierMask = coarseModel.getCoarse(fgMask)
                    
                    if bestPara is None : 
                        break
                    bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
                    flowCoarse = warper.warp_grid(bestPara)
                    
                    flowFine, matchFine, flowFineDown8, matchFineDown8 = PredFlowMask(coarseModel.IsTensor, featt, flowCoarse, grid, network)
                
                    
                    flowFinePlus = flowFine
                    matchFinePlus = matchFine
                    flowFinePlusDown8 = flowFineDown8
                    matchFinePlusDown8 = matchFineDown8
                    coarsePlusParam = bestPara
                        
                    
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
                    
                    np.save(os.path.join(outSceneFine, 'maskBG_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), It_bg.astype(bool))
                    
                    
                    np.save(os.path.join(outSceneFine, 'mask_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Fine_Mask_Tensor)
                    
                    
                    np.save(os.path.join(outSceneCoarse, 'flow_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Coarse_Flow_Tensor)
                    np.save(os.path.join(outSceneFine, 'flow_' + str(i) + '_{:d}H.npy'.format(nbCoarse)), Fine_Flow_Tensor)
                    
                    
                with open(outRotation, 'w') as f :
                    json.dump(angle_rotation, f)
            
                
                

