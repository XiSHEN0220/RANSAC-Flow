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
import pandas as pd 

def alignmentError(wB, hB, wA, hA, XA, YA, XB, YB, flow, match2, pixelGrid) : 
    estimX = flow.narrow(3, 1, 1).view(1, 1, hB, wB)
    estimY = flow.narrow(3, 0, 1).view(1, 1, hB, wB)

    estimY = ((estimY + 1) * 0.5 * (wA - 1))
    estimX = ((estimX + 1) * 0.5 * (hA - 1))
    match = match2.squeeze().numpy()
    estimY = estimY.squeeze().numpy()
    estimX = estimX.squeeze().numpy()
    
    xa, ya, xb, yb = XA.astype(np.int64), YA.astype(np.int64), XB.astype(np.int64), YB.astype(np.int64)
    index = np.where(match[yb, xb] > 0.5)[0]
    nbAlign = len(index)
    if nbAlign > 0 : 
        xa, ya, xb, yb = xa[index], ya[index], xb[index], yb[index]
        xaH = estimY[yb, xb]
        yaH = estimX[yb, xb]
        pixelDiff = ((xaH - xa) ** 2 + (yaH - ya) ** 2)**0.5
        pixelDiffT = pixelDiff.reshape((-1, 1))
        pixelDiffT = np.sum(pixelDiffT <= pixelGrid, axis = 0)
    else : 
        pixelDiffT = np.zeros(pixelGrid.shape[1])
    
    return pixelDiffT, nbAlign

## resize image according to the minsize, at the same time resize the x,y coordinate
def ResizeMinResolution(minSize, I, x, y, strideNet) : 

        x = np.array(list(map(float, x.split(';')))).astype(np.float32)
        y = np.array(list(map(float, y.split(';')))).astype(np.float32)
        
        w, h = I.size
        ratio = min(w / float(minSize), h / float(minSize)) 
        new_w, new_h = round(w/ ratio), round(h / ratio) 
        new_w, new_h = new_w // strideNet * strideNet , new_h // strideNet * strideNet
        
        ratioW, ratioH = new_w / float(w), new_h / float(h)
        I = I.resize((new_w, new_h), resample=Image.LANCZOS)
        
        x, y = x * ratioW, y * ratioH
        
        return I, x, y

## resize image according to the minsize, at the same time resize the x,y coordinate
def ResizeMinResolution_megadepth(minSize, I, x, y, strideNet) : 

        x = np.array(list(map(float, x.split(';')))).astype(np.float32)
        y = np.array(list(map(float, y.split(';')))).astype(np.float32)
        
        w, h = I.size
        ratio = min(w / float(minSize), h / float(minSize)) 
        new_w, new_h = round(w/ ratio), round(h / ratio) 
        new_w, new_h = new_w // strideNet * strideNet , new_h // strideNet * strideNet
        
        ratioW, ratioH = new_w / float(w), new_h / float(h)
        I = I.resize((new_w, new_h), resample=Image.LANCZOS)
        
        x, y = x * ratioW, y * ratioH
        index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)
        
        return I, x, y, index_valid                
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
    
    match = torch.from_numpy(match)
    match = F.interpolate(input = match, scale_factor = 8, mode='bilinear')
    
    match = match.narrow(1, 0, 1) * F.grid_sample(match.narrow(1, 1, 1), flowUp) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2) 
    #match = match.narrow(1, 0, 1) * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1)).type(torch.FloatTensor)).permute(0, 3, 1, 2)
    
    
    match = match.permute(0, 2, 3, 1)
    flow = torch.clamp(flow, min=-1, max=1)  
    flowGlobal = flow[:1]
    match_binary = match[:1] >= th
    matchGlobal = match[:1]
    if multiH : 
        
        for i in range(1, len(match)) : 
            tmp_match = (match.narrow(0, i, 1) >= th) * (~ match_binary)
            matchGlobal[tmp_match] = match.narrow(0, i, 1)[tmp_match]
            match_binary = match_binary + tmp_match 
            tmp_match = tmp_match.expand_as(flowGlobal)
            flowGlobal[tmp_match] = flow.narrow(0, i, 1)[tmp_match]
            
        
    
    return flowGlobal, matchGlobal
        
def getFlow_Coarse(pairID, flowList, finePath, coarsePath) :
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
    
    
    h, w = flow.size()[2], flow.size()[3]
    
    #### -- grid     
    gridY = torch.linspace(-1, 1, steps = h * 8).view(1, -1, 1, 1).expand(1, h * 8,  w * 8, 1)
    gridX = torch.linspace(-1, 1, steps = w * 8).view(1, 1, -1, 1).expand(1, h * 8,  w * 8, 1)
    grid = torch.cat((gridX, gridY), dim=3)
    
    warper = tgm.HomographyWarper(h * 8,  w * 8)
    
    coarse = warper.warp_grid(param.narrow(0, 0, 1))
    
    return coarse, torch.ones(1, h * 8,  w * 8, 1)


parser = argparse.ArgumentParser()


## model parameters


parser.add_argument('--multiH', action='store_true', help='multiple homograhy or not')

parser.add_argument('--onlyCoarse', action='store_true', help='only Coarse')

parser.add_argument('--minSize', type=int, default = 480, help='min size')

parser.add_argument('--matchabilityTH',type=float, nargs='+', default = [0], help='matchability threshold list')

parser.add_argument('--coarsePth', type=str, help='prediction file coarse flow ')

parser.add_argument('--finePth', type=str, help='prediction file fine flow')

parser.add_argument('--maskPth', type=str, help='prediction file mask')

parser.add_argument('--th', type=float, default=0.95, help='threshold')

parser.add_argument('--dataset', type=str, default='MegaDepth', help='RobotCar or megadepth')


subparsers = parser.add_subparsers(title="test dataset", dest="subcommand")


robotCar = subparsers.add_parser("RobotCar", help="parser for training arguments")

## test file
robotCar.add_argument('--testDir', type=str, default = '../../data/RobotCar/imgs/', help='RGB image directory')
robotCar.add_argument('--testCSV', type=str, default = '../../data/RobotCar/test6511.csv', help='RGB image directory')

megaDepth1600 = subparsers.add_parser("MegaDepth", help="parser for training arguments")

## test file
megaDepth1600.add_argument('--testDir', type=str, default = '../../data/MegaDepth/Test/test1600Pairs', help='RGB image directory')
megaDepth1600.add_argument('--testCSV', type=str, default = '../../data/MegaDepth/Test/test1600Pairs.csv', help='RGB image directory')
megaDepth1600.add_argument('--beginIndex', type=int, default = 0, help='begin index')
megaDepth1600.add_argument('--endIndex', type=int, default = 1600, help='end index')

args = parser.parse_args()


args = parser.parse_args()
print (args)


minSize = args.minSize
strideNet = 16

## Loading data    
# Set up for real validation
df = pd.read_csv(args.testCSV, dtype=str)

precAllAlign = {}
validAlign = {}

for th in args.matchabilityTH : 
    precAllAlign[th] = np.zeros(8)
    validAlign[th] = 0

pixelGrid = np.around(np.logspace(0, np.log10(36), 8).reshape(-1, 8))

print ('Evaluation for pixel grid : \n')
print ('-->  ', pixelGrid, '\n')


nbImg = len(df)
flowList = os.listdir(args.finePth)

for i in tqdm(range(nbImg)) : 
        
    scene = df['scene'][i]
    #### --  Source Image feature
    Is = Image.open( os.path.join( os.path.join(args.testDir, scene), df['source_image'][i]) ).convert('RGB') if scene != '/' else Image.open( os.path.join( args.testDir, df['source_image'][i]) ).convert('RGB')
    
    if args.dataset == 'RobotCar' : 
        Is, Xs, Ys = ResizeMinResolution(args.minSize, Is, df['XA'][i], df['YA'][i], strideNet)
        Isw, Ish = Is.size
        

        #### -- Target Image feature
        
        It = Image.open( os.path.join( os.path.join(args.testDir, scene), df['target_image'][i]) ).convert('RGB') if scene != '/' else Image.open( os.path.join( args.testDir, df['target_image'][i]) ).convert('RGB')
        It, Xt, Yt = ResizeMinResolution(args.minSize, It, df['XB'][i], df['YB'][i], strideNet)
    else : 
        Is, Xs, Ys, valids = ResizeMinResolution_megadepth(args.minSize, Is, df['XA'][i], df['YA'][i], strideNet)
        Isw, Ish = Is.size
        

        #### -- Target Image feature
        
        It = Image.open( os.path.join( os.path.join(args.testDir, scene), df['target_image'][i]) ).convert('RGB') if scene != '/' else Image.open( os.path.join( args.testDir, df['target_image'][i]) ).convert('RGB')
        It, Xt, Yt, validt = ResizeMinResolution_megadepth(args.minSize, It, df['XB'][i], df['YB'][i], strideNet)
        
        index_valid = valids * validt
        Xs, Ys, Xt, Yt = Xs[index_valid], Ys[index_valid], Xt[index_valid], Yt[index_valid]
    
    Itw, Ith = It.size
    flow, match = getFlow_Coarse(i, flowList, args.finePth, args.coarsePth) if args.onlyCoarse else getFlow(i, args.finePth, flowList, args.coarsePth, args.maskPth, args.multiH, args.th)
    
    if len(flow) == 0 :
        
         precAllAlign[0] = precAllAlign[th] + np.zeros(8)
         validAlign[0] += len(Xs)
         continue
            
    for th in args.matchabilityTH : 
            
        matchTH =  (match >= th ).type(torch.FloatTensor)
        matchabilityBinary = matchTH * (((flow.narrow(3, 0, 1) >= -1) * ( flow.narrow(3, 0, 1) <= 1)).type(torch.FloatTensor) * ((flow.narrow(3, 1, 1) >= -1) * ( flow.narrow(3, 1, 1) <= 1))).permute(0, 3, 1, 2).type(torch.FloatTensor) if th > 0 else torch.ones(match.size())
        pixelDiffT, nbAlign = alignmentError(Itw, Ith, Isw, Ish, Xs, Ys, Xt, Yt, flow, matchabilityBinary, pixelGrid)
        precAllAlign[th] = precAllAlign[th] + pixelDiffT
        validAlign[th] += nbAlign
        
for th in args.matchabilityTH  : 
    msg = '\nthreshold {:.1f}, precision '.format(th)
    print (msg, precAllAlign[th] / validAlign[th], validAlign[th])

    



        


