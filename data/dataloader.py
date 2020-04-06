
import os
import torch
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as data
import PIL.Image as Image
from torchvision import transforms


## This random crop is important and make the training data easy to learn
def resizeImg(I, minSize = 256) :

    w, h = I.size
    ## resize img, the smallest dimension is minSize
    wratio, hratio = w / minSize, h / minSize
    resizeRatio = min(wratio, hratio)
    w, h= int(round(w / resizeRatio) // 8 * 8), int(round(h / resizeRatio // 8 * 8))
    
    return I.resize((w, h), resample=Image.LANCZOS)    
                                    
def trainTransform(I1, I2, crop=224) : 
    resize = np.random.choice([crop, crop + crop // 2, crop  * 2])
    I1 = resizeImg(I1, minSize = resize)
    I2 = resizeImg(I2, minSize = resize)
    w, h = I1.size
    idw, idh = np.random.randint(w - crop) if w > crop else 0, np.random.randint(h - crop) if h > crop else 0   
    I1, I2 = I1.crop((idw, idh, idw + crop, idh + crop)), I2.crop((idw, idh, idw + crop, idh + crop))
    if np.random.rand() >= 0.5 : 
        I1 = I1.transpose(Image.FLIP_LEFT_RIGHT)
        I2 = I2.transpose(Image.FLIP_LEFT_RIGHT)
        
        
    I1, I2 = transforms.ToTensor()(I1), transforms.ToTensor()(I2)
    return I1, I2


                                   
def LoadImg(path) :
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, imgDir, dataTransform, imgSize, isTrain):
        self.imgDir = imgDir
        sample3 = os.path.join(self.imgDir,  '1_3.jpg')
        self.cycle = 3 if os.path.exists(sample3) else 2
        self.nbImg = len(os.listdir(self.imgDir) ) // self.cycle
        self.isTrain = isTrain
        self.dataTransform = dataTransform
        self.imgSize = imgSize
        
    def __getitem__(self, index):

        
        idx = np.random.choice(range(1, self.cycle + 1), 2, replace=False)
        path1 = os.path.join( self.imgDir,  '{:d}_{:d}.jpg'.format(index, idx[0]))
        path2 = os.path.join( self.imgDir,  '{:d}_{:d}.jpg'.format(index, idx[1]))
        
        I1, I2 = LoadImg(path1), LoadImg(path2)
        if self.isTrain : 
            I1, I2 = self.dataTransform(I1, I2, self.imgSize)
        else :     
            I1, I2 = self.dataTransform(I1), self.dataTransform(I2)
        
        return {'I1' : I1, 'I2' : I2}

    def __len__(self):
        return self.nbImg 

## Train Data loader
def TrainDataLoader(imgDir, trainT, batchSize, imgSize):

    trainSet = ImageFolder(imgDir, trainT, imgSize, isTrain = True)
    trainLoader = data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=1, drop_last = True)

    return trainLoader

## Val Data loader
def ValDataLoader(imgDir, valT, batchSize, imgSize):

    valSet = ImageFolder(imgDir, valT, imgSize, isTrain = False)
    valLoader = data.DataLoader(dataset=valSet, batch_size=batchSize, shuffle=False, num_workers=1, drop_last = True)

    return valLoader
