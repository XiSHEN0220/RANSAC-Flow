# Evaluation on sparse correspondences (MegaDepth and RobotCar )

### Data 


The KITTI 2015 dataset can be downloaded in [here](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow). We only use the data [stereo 2015/flow 2015/scene flow 2015 data set (2 GB)](http://www.cvlibs.net/download.php?file=data_scene_flow.zip). 

Once it is downloaded, saved it into `../../data`, the file structure should be : 
```
./RANSAC-Flow/data/Kitti
├── training/
│   ├── image_2/
│   ├── flow_noc/
│   └── flow_occ/
└── testing/
    ├── image_3/
    └── image_2/

```

### MOCO feature 

#### W/O Fine-tuning

Running :
 
``` Bash
python evaluation.py --outDir MOCO_WO_FT Kitti
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --predDir MOCO_WO_FT --resName Finetune --multiH --noc --interpolate
```

#### Fine-tuning

Running :
 
``` Bash
python evaluation.py --outDir MOCO_FT --resumePth ../../model/pretrained/KITTI_TestFT.pth Kitti
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --predDir MOCO_FT --resName Finetune --multiH --noc --interpolate
```


### ImageNet feature 

Adding `--imageNet` when running `evaluation.py` with the above commands.

