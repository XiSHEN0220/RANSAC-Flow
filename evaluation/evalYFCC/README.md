# Evaluation on Two-View Geometry Estimation

### Data 


We follow the evaluation of [OANet](https://github.com/zjhthu/OANet). Downloading the 4 test scenes used in [OANet](https://github.com/zjhthu/OANet#generate-training-and-testing-data) : buckingham_palace , notre_dame_front_facade, reichstag, sacre_coeur. 

Once it is downloaded, saved it into `../../data`, the file structure should be : 
```
./RANSAC-Flow/data/YFCC

├── images/
│   ├── buckingham_palace/
│   ├── notre_dame_front_facade/
│   ├── reichstag/
│   └── sacre_coeur/
└── pairs/
    ├── buckingham_palace-te-1000-pairs.pkl
    ├── notre_dame_front_facade-te-1000-pairs.pkl
    ├── reichstag-te-1000-pairs.pkl
    └── sacre_coeur-te-1000-pairs.pkl

```

### MOCO feature 

#### SegNet + W/O Fine-tuning

Running :
 
``` Bash
python evaluation.py --outDir MOCO --segNet YFCC 
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --multiH --ransac --coarsePth MOCO_Coarse --finePth MOCO_Fine --maskPth MOCO_Fine --outRes moco.json --scene 0
python getResults.py --multiH --ransac --coarsePth MOCO_Coarse --finePth MOCO_Fine --maskPth MOCO_Fine --outRes moco.json --scene 1
python getResults.py --multiH --ransac --coarsePth MOCO_Coarse --finePth MOCO_Fine --maskPth MOCO_Fine --outRes moco.json --scene 2
python getResults.py --multiH --ransac --coarsePth MOCO_Coarse --finePth MOCO_Fine --maskPth MOCO_Fine --outRes moco.json --scene 3

```


### ImageNet feature 

Adding `--imageNet` when running `evaluation.py` with the above commands.


### Metric
    
    According to the implementation of [OANet](https://github.com/zjhthu/OANet), the mAP@20 in the paper is the average over AP < 5, AP < 10, AP < 15, AP < 20; mAP@10 in the paper in the average over standard Acc < 5, Acc < 10. For more details, we refer to this part of code in [OANet](https://github.com/zjhthu/OANet/blob/master/core/test.py#L81). 
    
    Running our code will give you the AP < 5, AP < 10, AP < 15, AP < 20 (average over the 4 scenes). Then to compare to the numbers in the paper, one need to compute the average.
    
    We thank to [Jiahui Zhang](https://github.com/zjhthu) to point it out. 

