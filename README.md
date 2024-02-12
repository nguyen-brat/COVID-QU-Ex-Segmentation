# COVID-QU-Ex Segmentation
## Introduction
- We propose an end-to-end realtime system to detect, localize, and quantify COVID-19 infection from X-ray images.

## Proposed Architecture
<img src="./figs/model_architecture.png" alt="image" style="zoom:50%;" />

## Presentation Slide
- [AIO 2023 End Course Project Slide](https://docs.google.com/presentation/d/1m-uDndosn4_zgllyYscRTrMTSlkM1-KHUoAQVeejMCA/edit?usp=sharing)

## Experimental Results
|Task|Backbone|Accuracy|IoU|DSC|
|:------:|:------:|:------:|:------:|:------:|
|Lung Segmentation|MobileNet v3|98.09|92.05|95.77|
|Infection Segmentation|MobileNet v3|97.77|80.17|85.65|

**CPU running inference**: Intel(R) Xeon(R) CPU @ 2.20GHz <br>
**Inference time on average per image**: 0.02 s <br>
**Achieve realtime segmentation with 50 FPS** <br>
Fully code for training and reimplementing experimental results: [Kaggle Notebook](https://www.kaggle.com/code/khitrnhxun/final-model-quantization)

## Installation
```
pip install -r requirements.txt
```

## Streamlit App
[App Link](https://trinhxuankhai-covid-image-segmentation-main-3tf7sn.streamlit.app/) 

## Data Preparation

**COVID-QU-Ex Dataset:**  [Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) 

Organize the dataset as follows:
```
|- datasets
   |- Infection Segmentation Data
   |  |- Test
   |  |   |- COVID-19
   |  |   |- Non-COVID
   |  |   |- Normal
   |  |- Train
   |  |   |- COVID-19
   |  |   |- Non-COVID
   |  |   |- Normal
   |  |- Val
   |  |   |- COVID-19
   |  |   |- Non-COVID
   |  |   |- Normal
```
