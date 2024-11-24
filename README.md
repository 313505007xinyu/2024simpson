# 2024simpson
2024NYCU-ML-LAB2

## 1.做法說明
### environment:
透過MobaXterm連線到sever，並以Visual Studio Code作為操作界面，載入pytorch的image，建立pytorch的container
### Abstract:
本次競賽選擇以MobaXterm、Visual Studio Code作為開發環境來進行圖片辨識，將資料上傳，對資料集進行資料擴增，訓練模型辨識圖片。
透過相互比較各個模型的結果，挑選出最效果最好的模型，並透過調整模型和資料擴增的參數，來提升準確率。
#### 1.library與設備設定
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset, ConcatDataset
torch 和 torchvision 用於構建和訓練深度學習模型。
pandas 和 matplotlib.pyplot 用於處理數據和可視化。
torch.nn 和 torch.optim 用於構建神經網絡和設定優化器。

#### 2.設備設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
檢查 GPU 是否可用。如果有 CUDA 支持，會選擇 GPU 進行加速，否則使用 CPU。

#### 3.資料擴充
定義需要做的資料擴充，自定義4個transform(GaussianNoise、SpeckleNoise、PoissonNoise、SaltPepperNoise)，
接著定義三種不同的資料增強策略:原始資料增強、第一個資料增強策略（transformAug1）和第二種資料增強策略（transformAug2），
策略包括對圖片進行標準化、大小調整、隨機水平翻轉、隨機垂直翻轉、隨機旋轉、顏色調整、灰階、高斯模糊等。

#### 4.導入資料集(劃分驗證集與訓練集比例為2:8)
trainDataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transform)
aug1Dataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transformAug1)
aug2Dataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transformAug2)
combinDataset = ConcatDataset([trainDataset, aug1Dataset, aug2Dataset])
train_size = int(0.8 * len(combinDataset))
validation_size = len(combinDataset) - train_size
trainSet, validationSet = random_split(combinDataset, [train_size, validation_size])

#### 5.模型構建與訓練
model = models.efficientnet_v2_s().to(device)
model.classifier = nn.Linear(1280, num_classes).to(device)
使用了 EfficientNetV2_S 模型並修改了最後一層以適應當前分類任務的類別數。
訓練過程：對每個 epoch 執行正向傳播、損失計算、反向傳播、參數更新。
每隔 5 個 epoch 儲存一次。

#### 6.輸出結果
model.eval() 
for key, image in testImages.items():
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    testImages[key] = predictedLabel
    對測試集的每張圖像進行推理，並將結果儲存為 CSV 檔案。



