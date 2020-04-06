## Pretrained models

To download the models :
 
``` Bash
bash download_model.sh
```

You will download 8 models, including the segmentation network that we used to remove the sky in the 3D tasks : 

* *ade20k_resnet50dilated_encoder.pth* : encoder part of the pre-trained segmentation network [ResNet50dilated + PPM_deepsup](https://github.com/CSAILVision/semantic-segmentation-pytorch)

* *ade20k_resnet50dilated_decoder.pth* : decoder part of the pre-trained segmentation network [ResNet50dilated + PPM_deepsup](https://github.com/CSAILVision/semantic-segmentation-pytorch)

Our models trained on the MegaDepth training set : 

* *MegaDepth_Theta1_Eta001_Grad0_0.807.pth* : trained on MegaDepth training set, all the results without fine-tuning in the paper is with this model.

* *MegaDepth_Theta1_Eta001_Grad1_0.774.pth* : trained on MegaDepth training set with regulazing the gradient of the predicted flow. We generate visual results with this model.


Our models fine-tuned on the test set : 

* *MegaDepth_TestFT.pth* : trained on MegaDepth then fine-tuned on the MegaDepth test set for the correspondences tasks.

* *RobotCar_TestFT.pth* : trained on MegaDepth then fine-tuned on the RobotCar test set for the correspondences tasks.

* *KITTI_TestFT.pth* : trained on MegaDepth then fine-tuned on the KITTI test set for the optical flow tasks.

Finally, we also provide Moco features of ResNet-50, which is originally available [here](https://github.com/bl0/moco)
