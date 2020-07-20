#!/usr/bin/python3

import argparse
import os
import numpy as np
import torch
import warnings
from torchvision import transforms
import torch.nn.functional as F
import itertools
import pandas as pd
import pickle 
import time

import sys 
sys.path.append('..')


import model.model as model
import model.ssimLoss as ssimLoss

import data.dataloader as dataloader
#import utils.monitor as monitor # easy to support monitor
import validation

import ssimLoss
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def computeLossMatchability(network, I, indexRoll, grid, maskMargin, args, ssim, LrLoss):

    f = F.normalize(network['netFeatCoarse'](I), p=2, dim=1)
    
    corr = network['netCorr'](f[indexRoll], f)
    
    finalGrad, final = model.predFlowCoarse(corr, network['netFlowCoarse'], grid)
    
    #corr = corr.dlambda_matchch()       
    match = model.predMatchability(corr, network['netMatch']) * maskMargin 
    
    matchCycle = F.grid_sample(match[indexRoll], final) * match
    
    
    ## cycle loss on flow
    
    flowC = F.grid_sample(final[indexRoll].permute(0, 3, 1, 2), final).permute(0, 2, 3, 1)
    
    lossCycle = torch.mean(torch.abs(flowC - grid), dim=3).unsqueeze(1) ## Dim : N * 1 * W * H
    
    lossCycle = torch.sum(lossCycle * matchCycle) / (torch.sum(matchCycle) + 0.001) 

    ## Reconstruction Loss 3 channels 
    IWarp = F.grid_sample(I, final)
    
    lossLr =  LrLoss(IWarp, I[indexRoll], matchCycle, args.margin, maskMargin, ssim) 
    
    ## matchability loss
    lossMatch = torch.sum(torch.abs(1 - matchCycle) * maskMargin) / (torch.sum(maskMargin) + 0.001) 
    
    lossGrad = torch.sum(finalGrad * (1 - matchCycle[:, :, :-1, :-1]) * maskMargin[:, :, :-1, :-1]) / (torch.sum((1 - matchCycle[:, :, :-1, :-1]) * maskMargin[:, :, :-1, :-1]) + 0.001) 
    
    #lossGrad = torch.mean(finalGrad)
    loss = lossLr  + args.mu_cycle * lossCycle + args.lambda_match * lossMatch + args.grad * lossGrad 
    return lossLr.item(), lossCycle.item(), lossMatch.item(), lossGrad.item(), loss      
    


def computeLossNoMatchability(network, I, indexRoll, grid, maskMargin, args, ssim, LrLoss):

    f = F.normalize(network['netFeatCoarse'](I), p=2, dim=1)
    corr = network['netCorr'](f[indexRoll], f)
    _, final = model.predFlowCoarse(corr, network['netFlowCoarse'], grid)
    
    flowC = F.grid_sample(final[indexRoll].permute(0, 3, 1, 2), final).permute(0, 2, 3, 1)
    ## cycle loss on flow
    lossCycle = torch.mean(torch.abs(flowC - grid), dim=3).unsqueeze(1) ## Dim : N * 1 * W * H
    
    lossCycle = torch.sum(lossCycle * maskMargin) / (torch.sum(maskMargin) + 0.001) 

    ## Reconstruction Loss 3 channels 
    IWarp = F.grid_sample(I, final)
    
    lossLr =  LrLoss(IWarp, I[indexRoll], maskMargin, args.margin, maskMargin, ssim) 
    
    ## matchability loss
    
    loss = lossLr  + args.mu_cycle * lossCycle 
    
    return lossLr.item(), lossCycle.item(), 0, 0, loss  
    
def computeGradLossNoMatchability(network, I, indexRoll, grid, maskMargin, args, ssim, LrLoss):

    f = F.normalize(network['netFeatCoarse'](I), p=2, dim=1)
    corr = network['netCorr'](f[indexRoll], f)
    finalGrad, final = model.predFlowCoarse(corr, network['netFlowCoarse'], grid)
    
    flowC = F.grid_sample(final[indexRoll].permute(0, 3, 1, 2), final).permute(0, 2, 3, 1)
    ## cycle loss on flow
    lossCycle = torch.mean(torch.abs(flowC - grid), dim=3).unsqueeze(1) ## Dim : N * 1 * W * H
    
    lossCycle = torch.sum(lossCycle * maskMargin) / (torch.sum(maskMargin) + 0.001) 

    ## Reconstruction Loss 3 channels 
    IWarp = F.grid_sample(I, final)
    
    lossLr =  LrLoss(IWarp, I[indexRoll], maskMargin, args.margin, maskMargin, ssim) 
    
    lossGrad = torch.sum(finalGrad * maskMargin[:, :, :-1, :-1]) / (torch.sum( maskMargin[:, :, :-1, :-1] ) + 0.001) 
    ## matchability loss
    
    loss = lossLr  + args.mu_cycle * lossCycle + args.grad * lossGrad
    
    return lossLr.item(), lossCycle.item(), 0, lossGrad, loss       
      
def run(args) :
    
    if not torch.cuda.is_available():
        raise RuntimeError("Not support cpu version currently...")

    torch.backends.cudnn.benchmark = True
    ## Visom Visualization
    #logger = monitor.Logger(args.outDir)
      

    # Define Networks
    network = {'netFeatCoarse' : model.FeatureExtractor(), 
               'netCorr'       : model.CorrNeigh(args.kernelSize),
               'netFlowCoarse' : model.NetFlowCoarse(args.kernelSize), 
               'netMatch'      : model.NetMatchability(args.kernelSize),
               }
               
        
    
    for key in list(network.keys()) : 
        network[key].cuda()
        typeData = torch.cuda.FloatTensor
    
    # Network initialization
    if args.resumePth:
        param = torch.load(args.resumePth)
        msg = 'Loading pretrained model from {}'.format(args.resumePth)
        print (msg)
        
        for key in list(param.keys()) :
            try :  
                network[key].load_state_dict( param[key] )
            except : 
                print ('{} and {} weight not compatible...'.format(key, key)) 
    
    # Optimizers & LR schedulers
    ## only learn the flow, stage 1 and 2
    if args.trainMode == 'flow' : 
        optimizer     = [torch.optim.Adam(itertools.chain(*[network['netFeatCoarse'].parameters(), 
                                                           network['netCorr'].parameters(),
                                                           network['netFlowCoarse'].parameters()]), lr=args.lr, betas=(0.5, 0.999))]
        
        LossFunction = computeLossNoMatchability
        trainModule = ['netFeatCoarse', 'netCorr', 'netFlowCoarse']
    
    ## learn jointly the flow and match, final stage
    if args.trainMode == 'flow+match' : 
        optimizer     = [torch.optim.Adam(itertools.chain(*[network['netFeatCoarse'].parameters(), 
                                                           network['netCorr'].parameters(),
                                                           network['netFlowCoarse'].parameters()]), lr=args.lr, betas=(0.5, 0.999)),
                         torch.optim.Adam(itertools.chain(*[network['netMatch'].parameters()]), lr=args.lr, betas=(0.5, 0.999))]
        
        LossFunction = computeLossMatchability
        trainModule = ['netFeatCoarse', 'netCorr', 'netFlowCoarse', 'netMatch']
    
    ## smoothing the flow, for the visual results, introduce less distortions (with args.grad > 0)
    if args.trainMode == 'grad' : 
    
        optimizer     = [torch.optim.Adam(itertools.chain(*[network['netFlowCoarse'].parameters()]), lr=args.lr, betas=(0.5, 0.999))]
        
        LossFunction = computeLossMatchability
        trainModule = ['netFlowCoarse']
     
    
        
    
    ## Size Bs * 1 * (imgSize - 2 * margin) * (imgSize - 2 * margin)
    maskMargin = torch.ones(args.batchSize * 2, 1, args.imgSize - 2 * args.margin, args.imgSize - 2 * args.margin).type(typeData)
    maskMargin = F.pad(maskMargin, (args.margin, args.margin, args.margin, args.margin), "constant", 0)
    
    
    ## SSIM Loss
    ssim =  ssimLoss.SSIM()
    LrLoss = model.SSIM
    

    if not os.path.exists(args.outDir) :
        os.mkdir(args.outDir)
    outNet = os.path.join(args.outDir, 'BestModel.pth')
    
    # Train data loader
    trainT = dataloader.trainTransform
    
    trainLoader = dataloader.TrainDataLoader(args.trainImgDir, trainT, args.batchSize, args.imgSize)
    
    # Set up for real validation
    df = pd.read_csv(args.valCSV, dtype=str) if args.valCSV else None
    
    if args.inPklCoarse : 
        with open(args.inPklCoarse, 'rb') as f : 
            inPklCoarse = pickle.load(f)
    
    ## define the grid
    gridY = torch.linspace(-1, 1, steps = args.imgSize).view(1, -1, 1, 1).expand(1, args.imgSize,  args.imgSize, 1)
    gridX = torch.linspace(-1, 1, steps = args.imgSize).view(1, 1, -1, 1).expand(1, args.imgSize,  args.imgSize, 1)
    grid = torch.cat((gridX, gridY), dim=3).cuda()
    
    
    ## define loss and validation criteria
    bestPrec = 0
    LastUpdate = 0    
    
    
    index = np.arange(args.batchSize * 2)
    indexRoll = np.roll(index, args.batchSize)
    
    index = torch.from_numpy(index).cuda()
    indexRoll = torch.from_numpy(indexRoll).cuda()
    
    
    ###### Standard Training ######
    for epoch in range(args.nEpochs):
        
        trainLossLr = 0
        trainLossCycle = 0
        trainLossMatch = 0
        trainLossGrad = 0
        
        ## switch related module to train
        for key in  list(network.keys()) : 
            network[key].eval()
        
        for key in trainModule : 
            network[key].train()
        
        for i, batch in enumerate(trainLoader):
            # Set model input
            
            I = torch.cat((batch['I1'].cuda(), batch['I2'].cuda()), dim=0)
            # Forward
            for sub_optimizer in optimizer : 
                sub_optimizer.zero_grad()
                       
            # feature map B * 256 * W * H
            
            lossLr, lossCycle, lossMatch, lossGrad, loss = LossFunction(network, I, indexRoll, grid, maskMargin, args, ssim, LrLoss)
            loss.backward()
            
            for sub_optimizer in optimizer : 
                sub_optimizer.step()
                
            # Save loss
            trainLossLr += lossLr
            trainLossCycle += lossCycle
            trainLossMatch += lossMatch
            trainLossGrad += lossGrad
            
            # Print information
            if i % 50 == 49 :
                msg = '\n{}\tEpoch {:d}, Batch {:d}, Lr Loss: {:.9f}, Cycle Loss : {:.9f}, Matchability Loss {:.9f}, Gradient Loss {:.9f}'.format(time.ctime(), epoch, i + 1, trainLossLr / (i + 1), trainLossCycle/ (i + 1), trainLossMatch / (i + 1), trainLossGrad / (i + 1))
                print (msg)


        ## if some annotations are available, use them to validate models, otherwise use reconstruction loss
        if df is not None: 
            precFine = validation.validation(df, args.valImgDir, inPklCoarse, network, args.trainMode)
        else : 
            precFine = np.zeros(8)
            
        # Save train loss for one epoch
        trainLossLr = trainLossLr / len(trainLoader)
        trainLossCycle = trainLossCycle / len(trainLoader)
        trainLossMatch = trainLossMatch / len(trainLoader)
        trainLossGrad = trainLossGrad / len(trainLoader)
        
        
        log_loss = {'epoch':epoch, 'trainLossLr' : trainLossLr, 'trainLossCycle' : trainLossCycle, 'trainLossMatch':trainLossMatch, 'trainLossGrad':trainLossGrad, 'valPrec@8' : precFine[4]}
        
        msg = '\n{} Last Update {:d}---> Epoch {:d}, Train Lr Loss : {:.9f}, Train Cycle Loss : {:.9f}, Train Match Loss : {:.9f}, Train Grad Loss : {:.9f}, valPrec@8 : {:.9f} (Best {:.9f})-----'.format(time.ctime(), LastUpdate, epoch, trainLossLr, trainLossCycle, trainLossMatch, trainLossGrad, precFine[4], bestPrec)
        print (msg)
        
        valPrecEpoch = precFine[4]
        
        if df is not None and valPrecEpoch > bestPrec :
            msg = '\n{}\t---> Epoch {:d}, VAL Prec@8 IMPROVED: {:.9f} -- > {:.9f}-----'.format(time.ctime(), epoch, bestPrec, valPrecEpoch)
            print (msg)
            bestPrec = valPrecEpoch
            pth = {}
            for key in list(network.keys()) : 
                pth[key] = network[key].state_dict()
               
            torch.save(pth, outNet)
            LastUpdate = epoch
            
        elif df is None and epoch % args.epochSaveModel == args.epochSaveModel - 1: 
            outNet = os.path.join(args.outDir, 'checkPoint_Epoch{:d}_Lr{:.3f}_Lf{:.5f}_Lm{:.5f}_Lg{:.5f}'.format(epoch, trainLossLr, trainLossCycle, trainLossMatch, trainLossGrad))
            pth = {}
            for key in list(network.keys()) : 
                pth[key] = network[key].state_dict()
                
            torch.save(pth, outNet)
            
            print ('Save model to {}'.format(outNet))

    if df is not None :             
        finalOut = os.path.join(args.outDir, 'BestModel@8_{:.3f}.pth'.format(bestPrec)) 
        cmd = 'mv {} {}'.format(outNet, finalOut)
        os.system(cmd)        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    
    parser.add_argument('--trainImgDir', type=str, default = '../data/MegaDepth/MegaDepth_Train/', help='RGB train image directory')
    
    parser.add_argument('--kernelSize', type=int, default = 7, help='Kernel Size coarse')
    
    parser.add_argument('--imgSize', type=int, default = 224, help='image Size')
    
    parser.add_argument('--batchSize', type=int, default = 16, help='Batch Size')
    
    parser.add_argument('--outDir', type=str, help='output model directory')
    parser.add_argument('--resumePth', type=str, help='Resume directory')
    
    parser.add_argument('--lambda-match', type=float, default=0.01, help='matchability loss')
    
    parser.add_argument('--mu-cycle', type=float, default=1.0, help='weight for cycle distance')
    
    parser.add_argument('--grad', type=float, default=1, help='weight for cycle distance')
    
    parser.add_argument('--trainMode', choices=['flow', 'flow+match', 'grad'], help='choosing module for training...')
    
    parser.add_argument('--margin', type=int, default = 88, help='margin size, only training the central patch, in order to get rid of border effect')
    
    subparsers = parser.add_subparsers(title="validation choice", dest="subcommand")

    val = subparsers.add_parser("valMegaDepth", help="parser for training arguments")
    
    val.add_argument('--valImgDir', type=str, default='../data/MegaDepth/Val/img/', help='RGB val image directory')
    
    val.add_argument('--valCSV', type=str, default='../data/MegaDepth/Val/corr.csv', help='val csv containing correspondences')
    
    val.add_argument('--inPklCoarse', type=str, default='../data/MegaDepth/Val/coarse.pkl', help='input Coarse transformation to make model comparable')
    
    noval = subparsers.add_parser("NoVal", help="parser for training arguments")
    
    noval.add_argument('--valImgDir', type=str, help='RGB val image directory')
    
    noval.add_argument('--valCSV', type=str, help='val csv')
    
    noval.add_argument('--inPklCoarse', type=str, help='input Affine transformation to make model comparable')
    
    noval.add_argument('--epochSaveModel', type=int, help='# epochs save one model')
    
    args = parser.parse_args()
    
    print(args)
    
    if 'match' not in args.trainMode : 
        args.lambda_match = 0
        print ('\nChoose to not train with matchability, switch lambda_match to 0 for training and validation loss...\n')
        
    
    if args.trainMode == 'flow' : 
        print ('Train flow...')
    
    if args.trainMode == 'flow+match' : 
        print ('Train flow + matchability, train everything together...')
        
    if args.trainMode == 'grad' : 
        print ('Only learn to smooth flow without touching feature extractor + matchability net...')
    
    if args.resumePth is not None and 'NEED_TO_UPLOAD_CHECKPOINT' in args.resumePth : 
        print (args.resumePth)
        raise RuntimeError('{}'.format(args.resumePth.replace('_', ' '))) 

    run(args)
