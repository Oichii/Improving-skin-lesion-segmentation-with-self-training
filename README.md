# Improving Skin Lesion Segmentation with Self-Training
This repository is the official implementation of Improving Skin Lesion Segmentation with Self-Training 

## Abstract
Skin lesion segmentation plays a key role in the diagnosis of skin cancer; it can be a
component in both traditional algorithms and end-to-end approaches. The quality of segmentation
directly impacts the accuracy of classification; however, attaining optimal segmentation necessitates a
substantial amount of labeled data. Semi-supervised learning allows for employing unlabeled data
to enhance the results of the machine learning model. In the case of medical image segmentation,
acquiring detailed annotation is time-consuming and costly and requires skilled individuals so the
utilization of unlabeled data allows for a significant mitigation of manual segmentation efforts. This
study proposes a novel approach to semi-supervised skin lesion segmentation using self-training
with a Noisy Student. This approach allows for utilizing large amounts of available unlabeled images.
It consists of four steps — first, training the teacher model on labeled data only, then generating
pseudo-labels with the teacher model, training the student model on both labeled and pseudo-labeled
data, and lastly, training the student* model on pseudo-labels generated with the student model. In
this work, we implemented DeepLabV3 architecture as both teacher and student models. As a final
result, we achieved a mIoU of 88.0% on the ISIC 2018 dataset and a mIoU of 87.54% on the PH2
dataset. The evaluation of the proposed approach shows that Noisy Student training improves the
segmentation performance of neural networks in a skin lesion segmentation task while using only
small amounts of labeled data.

## Datasets
We conduct experiments on two public skin lesion datasets ISIC 2017 + 2018 and PH2.
data can be downloaded from 
* ISIC: https://challenge2020.isic-archive.com/
* PH2: https://www.fc.up.pt/addi/ph2%20database.html

## Requirements 
* python 3.7 
* PyTorch 1.9.1
* Segmentation Models Pytorch 0.2.1
* torch metrics 0.9.3
* Albumentations 1.1.0

## Training
To run the code update the paths in `config_segmentation.json` and run: 
```
python main_train_segmentation.py
```
