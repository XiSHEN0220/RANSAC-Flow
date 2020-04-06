# Evaluation on Hpatches

### Data 

We follow the evaluation provided by [DGC-Net](https://github.com/AaltoVision/DGC-Net).

Hpatches data can be downloaded from [here](https://github.com/hpatches/hpatches-dataset) at the end of page, or clicking [here](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz). 

The csv files containing the ground-truth are provided in the repo of [DGC-Net](https://github.com/AaltoVision/DGC-Net), more precisely, are accessible [here](https://github.com/AaltoVision/DGC-Net/tree/master/data/csv).

Once everything is downloaded, saved it into `../../data`, the file structure should be : 
```
./RANSAC-Flow/data/Hpatch
├── csv/
└── hpatches-sequences-release/
```

### MOCO feature 

Running :
 
``` Bash
python evaluation.py  --outDir MOCO_WO_FT --transformation Homography --maxCoarse 0
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --coarsePth MOCO_WO_FT_Coarse --finePth MOCO_WO_FT_Fine
```


To get results with only **coarse alignment** : 

``` Bash
python getResults.py --coarsePth MOCO_WO_FT_Coarse --finePth MOCO_WO_FT_Fine --onlyCoarse
```



### ImageNet feature 

Running :
 
``` Bash
python evaluation.py --outDir ImageNet_WO_FT --transformation Homography --maxCoarse 0 --imageNet
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --coarsePth ImageNet_WO_FT_Coarse --finePth ImageNet_WO_FT_Fine
```


To get results with only **coarse alignment** : 

``` Bash
python getResults.py --coarsePth ImageNet_WO_FT_Coarse --finePth ImageNet_WO_FT_Fine --onlyCoarse
```






