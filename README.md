# Kaggle_RSNA2020_6th_Solution

This is repository of the 6th place solution of [kaggle RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection).  
The discription of this solution is available [here]().  
The prediction notebook in the competition is available [here](https://www.kaggle.com/osciiart/rsna2020-final-sub2?scriptVersionId=45543346).  
# Enviroment
Google Cloud Platform
- SMP Debian 4.9.210-1 (2020-01-20)
- n1-standard-16 (vCPU x 16, memory 60 GB)
- 1 x NVIDIA Tesla P100

# Requirements
- Python 3.7.6
- CUDA 10.1
- cuddn 7.6.3
- gdcm 2.8.9
- numpy==1.19.1
- pandas==1.1.1
- matplotlib==3.2.1
- opencv-python==4.3.0.36
- pydicom==2.0.0
- scikit-learn==0.23.1
- torch==1.6.0+cu101
- torchvision==0.7.0+cu101
- timm==0.1.26
- albumentations==0.4.5  

# Data setup
Download the [competition dataset](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data) and place them in `input/orig/`.  
In case you use pretrained weights, download [the weights](https://www.kaggle.com/osciiart/rsna2020-pretrained-weights) and place them to `models/`.  

# Training and Prediction
Run all the cells of `notebook/preprocess.ipynb`.  
Run all the cells of `notebook/train_stage1.ipynb`.  
run all cells of `notebook/train_stage2.ipynb`.  
Rewrite the line 9 of the 5th cell of `notebook/train_stage1.ipynb` to `MODEL_NAME = 'b2'` and run all the cells.  
Rewrite the line 8 of the 5th cell of `notebook/train_stage1.ipynb` to `MODEL_NAME = 'b2'` and run all the cells.  
Run all the cells of `notebook/postprocess.ipynb`.  
Run all the cells of `notebook/predict.ipynb`.  

# Prediction with the pretrained weights
Rewrite the line 12-15 of the 3rd cell of `notebook/predict.ipynb` to  
```
weight_dir_b0_1 = "../models/b0_stage1"
weight_dir_b0_2 = "../models/b0_stage2"
weight_dir_b2_1 = "../models/b2_stage1"
weight_dir_b2_2 = "../models/b2_stage2"
```
and run all the cells.  