import segModel
import segData
import torch 
import numpy as np

class SegNet:
    
    def __init__(self, encoderPth, decoderPth, segId = 1, segFg = True):
        
        ## nb iteration, tolerance, transform
        ## 1 is building, 2 is sky
        net_encoder = segModel.ModelBuilder.build_encoder(fc_dim=2048, weights=encoderPth)
        net_decoder = segModel.ModelBuilder.build_decoder(fc_dim=2048, num_class=150, weights=decoderPth)

        self.net = segModel.SegmentationModule(net_encoder, net_decoder)
        self.net.eval()
        self.net.cuda()

        self.dataset_test = segData.TestDataset(imgSizes=(300, 375, 450, 525, 600), imgMaxSize = 500, padding_constant = 8)
        self.segId = segId
        self.segFg = segFg
         
    def getSky(self, imgPath) : 
        
        I_Tensor = self.dataset_test.getImg(imgPath)
        
        
        with torch.no_grad():    
            segSize = (I_Tensor['img_ori'].shape[0], I_Tensor['img_ori'].shape[1])
            scores = torch.zeros(1, 150, segSize[0], segSize[1]).cuda()
            
            for img in I_Tensor['img_data']:
                
                # forward pass
                pred_tmp = self.net(img.cuda(), segSize=segSize)
                scores = scores + pred_tmp / 5

            _, pred = torch.max(scores, dim=1)
            pred = pred.squeeze(0).cpu().numpy()
            if self.segFg :
                return (1 - (pred == self.segId).astype(np.float32))
            else : 
                return (pred == self.segId).astype(np.float32)
            
            
        

