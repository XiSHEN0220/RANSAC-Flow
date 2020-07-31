# Evaluation on sparse correspondences (MegaDepth and RobotCar )

### Data 

#### MegaDepth

We release our test MegaDepth dataset including 1600 pairs, can be downloaded here [here](https://drive.google.com/file/d/1SikcOvCJ-zznOyCRJCTGtpKtTp01Jx5g/view?usp=sharing).


Once it is downloaded, saved it into `../../data`, the file structure should be : 
```
./RANSAC-Flow/data/MegaDepth
├── MegaDepth_Train/
├── MegaDepth_Train_Org/
├── Val/
└── Test/
```

#### RobotCar

The RobotCar dataset can be downloaded from the [Visual Localization Challenge](https://www.visuallocalization.net/datasets/) (at the bottom of the site), or more precisely [here](https://www.dropbox.com/sh/ql8t2us433v8jej/AAB0wfFXs0CLPqSiyq0ukaKva/ROBOTCAR?dl=0&subfolder_nav_tracking=1). 

This evaluation code takes the same structure as MegaDepth, which means you need to have a csv containing all the correspondences and the image paths. The file structure should be : 

```
./RANSAC-Flow/data/RobotCar
├── img/
└── test6511.csv
```

The cvs file can be downloaded from [here(~5G)](https://drive.google.com/file/d/16mZLUKsjceAt1RTW1KLckX0uCR3O4x5Q/view?usp=sharing).


### MOCO feature 

#### MegaDepth, W/O Fine-tuning

Running :
 
``` Bash
python evaluation.py --outDir MOCO_MegaDepth_WO_FT MegaDepth
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --coarsePth MOCO_MegaDepth_WO_FT_Coarse --finePth MOCO_MegaDepth_WO_FT_Fine --dataset MegaDepth --multhH MegaDepth
```

#### MegaDepth, W Fine-tuning

Running :
 
``` Bash
python evaluation.py --outDir MOCO_MegaDepth_FT --resumePth ../../model/pretrained/MegaDepth_TestFT.pth MegaDepth
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --coarsePth MOCO_MegaDepth_FT_Coarse --finePth MOCO_MegaDepth_FT_Fine --dataset MegaDepth --multhH MegaDepth
```



#### RobotCar, W/O Fine-tuning
Running :
 
``` Bash
python evaluation.py --outDir MOCO_RobotCar_WO_FT RobotCar
```

To get results with our **fine alignment** : 

``` Bash
python getResults.py --coarsePth MOCO_RobotCar_WO_FT_Coarse --finePth MOCO_RobotCar_WO_FT_Fine --multhH --dataset RobotCar RobotCar
```

#### RobotCar, With Fine-tuning
Running :
 
``` Bash
python evaluation.py --outDir MOCO_RobotCar_FT --resumePth ../../model/pretrained/RobotCar_TestFT.pth  RobotCar
```

To get results with our **fine alignment** : 

``` Bash 
python getResults.py --coarsePth MOCO_RobotCar_FT_Coarse --finePth MOCO_RobotCar_FT_Fine --multhH --dataset RobotCar RobotCar
```



### ImageNet feature 

Adding `--imageNet` when running `evaluation.py` with the above commands.

