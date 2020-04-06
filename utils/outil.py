import torch.nn.functional as F
import PIL.Image as Image 
import torch 
import numpy as np

def resizeImg(I, strideNet, minSize = 400, mode=Image.LANCZOS) :

    w, h = I.size
    ## resize img, the largest dimension is maxSize
    wratio, hratio = w / minSize, h / minSize
    resizeRatio = min(wratio, hratio)
    
    w, h= w / resizeRatio, h / resizeRatio
    
    resizeW = round(w/ strideNet) * strideNet
    resizeH = round(h/ strideNet) * strideNet
    
    
    return I.resize((resizeW, resizeH), resample=mode)

def getWHTensor(feat):
    W = (torch.arange(0, feat.size(2), device=feat.device).view(-1, 1).expand(feat.size(2),feat.size(3)).contiguous().view(-1).float() + 0.5)/(feat.size(2))
    H = (torch.arange(0, feat.size(3), device=feat.device).view(1, -1).expand(feat.size(2),feat.size(3)).contiguous().view(-1).float() + 0.5)/(feat.size(3))
    return (W - 0.5) * 2, (H - 0.5) * 2
    
def getWHTensor_Int(feat):
    W = (torch.arange(0, feat.size(2), device=feat.device).view(-1, 1).expand(feat.size(2),feat.size(3)).contiguous().view(-1))
    H = (torch.arange(0, feat.size(3), device=feat.device).view(1, -1).expand(feat.size(2),feat.size(3)).contiguous().view(-1))
    return W, H
    

def mutualMatching(featA, featB) : 
    
    score = torch.mm(featA.transpose(0, 1), featB) #nbA * nbB
    
    maxDim0, maxDim0Index = score.topk(k=1, dim = 0) # 1 * nbB
    maxDim1, maxDim1Index = score.topk(k=1, dim = 1) # nbA * 1

    keepMaxDim0 = torch.zeros((featA.size(1), featB.size(1)), device=featA.device).scatter_(0, maxDim0Index, maxDim0)
    keepMaxDim1 = torch.zeros((featA.size(1), featB.size(1)), device=featA.device).scatter_(1, maxDim1Index, maxDim1)

    keepMax = keepMaxDim0 * keepMaxDim1
    keepMaxIndex = (keepMax > 0).nonzero()
    index1, index2 = keepMaxIndex[:, 0], keepMaxIndex[:, 1]
    return index1, index2
    
        
def Affine(X, Y):
    H21 = np.linalg.lstsq(Y, X[:, :2])[0]
    H21 = H21.T
    H21 = np.array([[H21[0, 0], H21[0, 1], H21[0, 2]], 
                    [H21[1, 0], H21[1, 1], H21[1, 2]],
                    [0, 0, 1]])
    
    return H21

def Hough(X, Y) : 
    nb_points = X.shape[0]
    ones = np.ones((nb_points, 1))
    H21x = np.linalg.lstsq(np.hstack((Y[:, 0].reshape((-1, 1)), ones)), X[:, 0])[0]
    H21y = np.linalg.lstsq(np.hstack((Y[:, 1].reshape((-1, 1)), ones)), X[:, 1])[0]

    H21 = np.array([[H21x[0], 0, H21x[1]],
                    [0, H21y[0], H21y[1]],
                    [0, 0, 1]])
    return H21

def Homography(X, Y):
    N = X.shape[0]
    device = X.device
    A = np.zeros((N, 8, 9))
    for i in range(4) :
        u, v, u_, v_ = Y[:, i, 0].cpu().numpy(), Y[:, i, 1].cpu().numpy(), X[:, i, 0].cpu().numpy(), X[:, i, 1].cpu().numpy()
        A[:, 2 * i] = np.stack([
            np.zeros(N), np.zeros(N), np.zeros(N),
            -u, -v, -np.ones(N), v_ * u, v_ * v, v_
        ], axis=1)
        A[:, 2 * i + 1] = np.stack([
            u, v, np.ones(N), np.zeros(N), np.zeros(N),
            np.zeros(N), -u_ * u, -u_ * v, -u_
        ], axis=1)

    #svd compositionq
    u, s, v = np.linalg.svd(A)
    #reshape the min singular value into a 3 by 3 matrix
    H21 = torch.tensor(np.reshape(v[:, 8], (N, 3, 3))).float().cuda()
    return H21

def Translation(X, Y) : 
    tx = X[0, 0] - Y[0, 0]
    ty = X[0, 1] - Y[0, 1]
    H21 = np.array([[1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]])
    return H21

def Prediction(X, Y, H21) :
    estimX = Y @ torch.transpose(H21, 1, 2)
    estimX = estimX / estimX[..., 2:]
    return torch.sum((X[..., :2] - estimX[..., :2])**2, dim=2)**0.5
    
def ScoreRANSAC(match1, match2, tolerance, samples, Transform) :
    #All The Data
    X = match1[samples]  # Niter * nbp * 3
    Y = match2[samples]

    H21 = Transform(X, Y)
    dets = torch.det(H21)
    
    error = Prediction(match1.unsqueeze(0), match2.unsqueeze(0), H21)
    isInlier = error < tolerance

    return H21, torch.sum(isInlier, dim=1) * (dets > 1e-6).long()



def RANSAC(nbIter, match1, match2, tolerance, nbPoint, Transform) :
    nbMatch = len(match1)
    
    samples = torch.randint(nbMatch, (nbIter, nbPoint), device=match1.device)

    # HARDCODED FOR HOMOGRPAHIES FOR NOW
    conditions = torch.stack([
        samples[:, 0] == samples[:, 1],
        samples[:, 0] == samples[:, 2],
        samples[:, 0] == samples[:, 3],
        samples[:, 1] == samples[:, 2],
        samples[:, 1] == samples[:, 3],
        samples[:, 2] == samples[:, 3]
    ], dim=1)  # N * nb_cond

    duplicated_samples = torch.any(conditions, dim=1)
    unique_samples = samples[~duplicated_samples] # N * nbPoint

    ## set max iter to avoid memory issue
    nbMaxIter = 100
    nbLoop = len(unique_samples) // nbMaxIter
    bestParams, bestInlier, isInlier, match2Inlier = None, 0, [], []
    
    for i in range(nbLoop) : 
        H21, nbInlier = ScoreRANSAC(match1, match2, tolerance, unique_samples.narrow(0, i * nbMaxIter, nbMaxIter), Transform)

        best = torch.argmax(nbInlier)
        
        if nbInlier[best] == 0:
            return None, 0, [], []

        elif nbInlier[best] > bestInlier:
            bestParams = H21[best]
            bestInlier = nbInlier[best]
        
    
    if len(unique_samples) - nbLoop * nbMaxIter > 0 :
        H21, nbInlier = ScoreRANSAC(match1, match2, tolerance, unique_samples.narrow(0, nbLoop * nbMaxIter, len(unique_samples) - nbLoop * nbMaxIter), Transform)

        best = torch.argmax(nbInlier)
    
        if nbInlier[best] > bestInlier:
           bestParams = H21[best]
           bestInlier = nbInlier[best]
            
    error = Prediction(match1[None], match2[None], bestParams[None]).squeeze(0)
    isInlier = error < tolerance
    return bestParams.cpu().numpy(), bestInlier.cpu().numpy(), isInlier.cpu().numpy(), match2[isInlier].cpu().numpy()

    
def SaliencyCoef(feat) : 
    b,s,w,h=feat.size()
    tmpFeat = F.pad(feat, (1, 1, 1, 1), 'reflect')
    tmpCoef =torch.cat([ torch.sum(tmpFeat.narrow(start=2, dim=2, length=w).narrow(start=1, dim=3, length=h) * feat, dim = 1, keepdim=True),
                         torch.sum(tmpFeat.narrow(start=0, dim=2, length=w).narrow(start=1, dim=3, length=h) * feat, dim = 1, keepdim=True),
                         torch.sum(tmpFeat.narrow(start=1, dim=2, length=w).narrow(start=0, dim=3, length=h) * feat, dim = 1, keepdim=True), 
                         torch.sum(tmpFeat.narrow(start=1, dim=2, length=w).narrow(start=2, dim=3, length=h) * feat, dim = 1, keepdim=True) ] , dim = 1)
    
    tmpCoef = torch.mean(tmpCoef, dim=1, keepdim=True)
    return tmpCoef
