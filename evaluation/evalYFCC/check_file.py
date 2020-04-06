import argparse
import os 
import json 
import pickle 
parser = argparse.ArgumentParser()

## model parameters


parser.add_argument('--coarsePth', type=str, help='prediction file coarse flow ')

parser.add_argument('--finePth', type=str, help='prediction file fine flow')

parser.add_argument('--maskPth', type=str, help='prediction file mask')

parser.add_argument('--gtPath', default = '../../data/YFCC/images/', type=str, help='ground truth file')

parser.add_argument('--testPair', type=str, default = '../../data/YFCC/pairs', help='RGB image directory')

parser.add_argument('--scene', type=int, choices=[0, 1, 2, 3], help='RGB image directory')
    
args = parser.parse_args()
print (args)


def getFlow(pairID, finePath, flowList, coarsePath, maskPath) :
    find = False 
    for flowName in flowList :
        if flowName.split('_')[1] == str(pairID) : 
            nbH = flowName.split('_')[2].split('H')[0]
            find = True
            break
            
    if not find : 
        return []
    else :
        print ('find {:d}'.format(pairID))    
    flow = os.path.join(finePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))
    param =os.path.join(coarsePath, 'flow_{:d}_{}H.npy'.format(pairID, nbH))
    match = os.path.join(finePath, 'mask_{:d}_{}H.npy'.format(pairID, nbH))
    matchBG = os.path.join(maskPath, 'maskBG_{:d}_{}H.npy'.format(pairID, nbH))
    
    if not (os.path.exists(flow) and os.path.exists(param) and os.path.exists(match) and os.path.exists(matchBG)) : 
        print (flow, param, match, matchBG)
        raise RuntimeError('XXX')
    return []
    
scene = ['notre_dame_front_facade', 'buckingham_palace', 'reichstag', 'sacre_coeur']   
test_scene = [scene[args.scene]]

for scene in test_scene :

    print ('evaluation on scene {}'.format(scene))    
    finePath = os.path.join(args.finePth, scene)
    coarsePath = os.path.join(args.coarsePth, scene)
    maskPath = os.path.join(args.maskPth, scene)
    flowList = [item for item in os.listdir(finePath) if 'flow' in item]
    rotation = os.path.join(args.maskPth, scene, 'rotation.json')
    with open(rotation, 'r') as f :
        rotation = json.load(f)
        
    with open(os.path.join(args.testPair, scene + '-te-1000-pairs.pkl'), 'rb') as f :
            pairs_ids = pickle.load(f)
    
    for i, (idA, idB) in enumerate(pairs_ids):
            
            
        ## read flow and matchability
        getFlow(i, finePath, flowList, coarsePath, maskPath)
        try : 
            rotation[str(i)]
        except : 
            raise RuntimeError('{}'.format(str(i)))
        
