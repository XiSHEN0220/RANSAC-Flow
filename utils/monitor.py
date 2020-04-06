from visdom import Visdom
import cv2
import numpy as np 

def tensor2image(tensor):

    if 'PIL' in str(type(tensor)) : 
        return np.array(tensor)
        
    image = 255*(tensor[0].cpu().float().numpy())
    
    if image.shape[0] == 1:
        image = cv2.applyColorMap(255 - image[0].astype(np.uint8), cv2.COLORMAP_JET)
        image = np.transpose(np.asarray(image), (2, 0, 1)) 
        image = image.astype(np.uint8)
        
    return image
    

class Logger():
    def __init__(self, env):
        self.viz = Visdom(env = env)
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        
        # End of epoch
        for loss_name, loss in losses.items() : 
            if loss_name == 'epoch' : 
                continue
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=np.array([0]), Y=np.array([loss]),
                                                                        opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name, 'height':256, 'width':256})
            else:
                self.viz.line(X=np.array([losses['epoch']]), Y=np.array([loss]), win=self.loss_windows[loss_name], update='append')
                
