import PIL.Image as Image 
import os 
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--inputDir', type=str, help='input directory')
parser.add_argument('--outputDir', type=str, help='output directory')
parser.add_argument('--maxSize', type=int, help='max image size')


args = parser.parse_args()
print (args)


def ResizeMaxSize(I, maxSize) : 

    w, h = I.size
    ratio = max(w / float(maxSize), h / float(maxSize)) 
    new_w, new_h = int(round(w/ ratio)), int(round(h / ratio)) 
    
    ratioW, ratioH = new_w / float(w), new_h / float(h)
    Iresize = I.resize((new_w, new_h), resample=Image.LANCZOS)
    
    return Iresize
    
if not os.path.exists(args.outputDir) : 
    os.mkdir(args.outputDir)
    
for i, img in enumerate(os.listdir(args.inputDir)) : 
    
    imgPth = os.path.join(args.inputDir, img)
    I = Image.open(imgPth).convert('RGB')
    I = ResizeMaxSize(I, args.maxSize)
    
    I.save(os.path.join(args.outputDir, '{}.png'.format(i)))
    
    
    
