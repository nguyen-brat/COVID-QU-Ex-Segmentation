# COVID-QU-Ex Segmentation

## Proposed Architecture
<img src="./figs/model_architecture.png" alt="image" style="zoom:50%;" />

## Installation
```
pip install -r requirements.txt
pip install openvino-dev[pytorch,onnx]
pip install numpy --upgrade
```
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