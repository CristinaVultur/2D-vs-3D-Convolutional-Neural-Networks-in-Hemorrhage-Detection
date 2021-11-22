# 2D-vs-3D-Convolutional-Neural-Networks-in-Hemorrhage-Detection
The paper studies the development and comparison of two convolutional neural network ResNet architectures, a 2D convolution and a 3D, for intracranial haemorrhage detection and classification problem in CT images. The paper can be found in the file "Vultur Cristina licenta", it has the details about the architectures chosen and implementation, the theory applied. Code detailed explanation can be found in Chapter 6, starting with page 42.

Has a GUI that used GRANDcam for looking at the heat signature where the model detects the hemorrage, and the predictions given for the 2D Model.

## Requirements
Needs MedicalNet repository found here:
https://github.com/Tencent/MedicalNet

And Kaggle Dataset:
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data

## ResNet Models

2DCNN Pretrained ResNet18
![image](https://user-images.githubusercontent.com/48892856/142818658-73119cea-3bfa-4950-a54b-f8ba1c780c72.png)


3DCNN Pretrained ResNet18 
- using the whole scan

![image](https://user-images.githubusercontent.com/48892856/142818739-7d8ae1a4-6c43-4e22-b9b7-e7ea001ddb43.png)

- using neighboring slices

![Untitled Diagram](https://user-images.githubusercontent.com/48892856/142821671-a7e7f9e0-049e-44e0-995b-3edc785d5538.jpg)


## Ciation

    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }

