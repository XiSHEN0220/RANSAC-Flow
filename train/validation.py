
import torch
import numpy as np
from tqdm import tqdm
import sys 
sys.path.append('..')


import model.model as model
import torch.nn.functional as F
import PIL.Image as Image
import os 
from torchvision import transforms
    
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
        
def alignmentError(wB, hB, wA, hA, XA, YA, XB, YB, flow, pixelGrid) : 
    estimX = flow.narrow(3, 1, 1).view(1, 1, hB, wB)
    estimY = flow.narrow(3, 0, 1).view(1, 1, hB, wB)

    estimY = ((estimY + 1) * 0.5 * (wA - 1))
    estimX = ((estimX + 1) * 0.5 * (hA - 1))
    
    pixelDiff = []
    for j in range(len(XB)) : 
        xa, ya, xb, yb = int(XA[j]), int(YA[j]), int(XB[j]), int(YB[j])
        xaH = estimY[0, 0, yb, xb].item()
        yaH = estimX[0, 0, yb, xb].item()
        pixelDiff.append( ((xaH - xa) ** 2 + (yaH - ya) ** 2)**0.5 )
            
        
    nbAlign = len(pixelDiff)
    
    pixelDiffT = np.array(pixelDiff).reshape((-1, 1))
    pixelDiffT = np.sum(pixelDiffT < pixelGrid, axis = 0)
    
    return pixelDiffT, nbAlign
    


def validation(df, valDir, inPklCoarse, network, trainMode) : 

    strideNet = 16
    minSize=480
    precAllAlign = np.zeros(8)
    totalAlign = 0
    pixelGrid = np.around(np.logspace(0, np.log10(36), 8).reshape(-1, 8))

    for key in list(network.keys()) : 
        network[key].eval()
        
    with torch.no_grad() : 
        
        for i in tqdm(range(len(df))) : 
            
            scene = df['scene'][i]
            
            #### --  Source Image feature
            Is = Image.open( os.path.join( os.path.join(valDir, scene), df['source_image'][i]) ).convert('RGB') 
            
            Is, Xs, Ys = ResizeMinResolution(minSize, Is, df['XA'][i], df['YA'][i], strideNet)
            
            Isw, Ish = Is.size
            IsTensor = transforms.ToTensor()(Is).unsqueeze(0).cuda()
            
            
        
            #### -- Target Image feature
            
            It = Image.open( os.path.join( os.path.join(valDir, scene), df['target_image'][i]) ).convert('RGB') 
            
            It, Xt, Yt = ResizeMinResolution(minSize, It, df['XB'][i], df['YB'][i], strideNet)
            Itw, Ith = It.size
            ItTensor = transforms.ToTensor()(It).unsqueeze(0).cuda()
            
            #### -- grid     
            gridY = torch.linspace(-1, 1, steps = ItTensor.size(2)).view(1, -1, 1, 1).expand(1, ItTensor.size(2),  ItTensor.size(3), 1)
            gridX = torch.linspace(-1, 1, steps = ItTensor.size(3)).view(1, 1, -1, 1).expand(1, ItTensor.size(2),  ItTensor.size(3), 1)
            grid = torch.cat((gridX, gridY), dim=3).cuda() 
            
            bestParam = inPklCoarse[i]
            flowGlobalT = F.affine_grid(torch.from_numpy(bestParam).unsqueeze(0).cuda(), ItTensor.size()) # theta should be of size N×2×3
            IsSample = F.grid_sample(IsTensor, flowGlobalT)
            
            featsSample = F.normalize(network['netFeatCoarse'](IsSample))
            featt = F.normalize(network['netFeatCoarse'](ItTensor))
            
            
            corr21 = network['netCorr'](featt, featsSample)
            _, flowCoarse =  model.predFlowCoarse(corr21, network['netFlowCoarse'], grid)
            flowFinal = F.grid_sample(flowGlobalT.permute(0, 3, 1, 2), flowCoarse).permute(0, 2, 3, 1).contiguous()
            
            pixelDiffT, nbAlign = alignmentError(Itw, Ith, Isw, Ish, Xs, Ys, Xt, Yt, flowFinal, pixelGrid)
            precAllAlign += pixelDiffT
            totalAlign += nbAlign
            
    return precAllAlign / totalAlign
    
