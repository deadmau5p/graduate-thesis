# Stance Detection in Slovenian News Media - [thesis](https://repozitorij.uni-lj.si/Dokument.php?id=166289&lang=slv)

Analyzing the bias in Slovenian news media regarding political-ideological topics and individuals frequently featured within. This project aims to categorize articles into classes (against, for, neutral) according to the authors' inclination towards a particular topic or individual. 

## 🚀 Abstract
The journey begins with the challenge of stance detection in Slovene language which is yet to be resolved due to the absence of a relevant dataset. Our venture utilizes a publicly accessible labeled dataset of Twitter posts in English, along with its translated Slovenian version to train models for this task. Two classification models based on BERT, namely SloBERTa and CroSloEngualBERT, are put to the test. The results exhibit considerable variations across different topics, with most models excelling on full articles. Feminism emerged as the top topic with an F1-measure of 0.58, while atheism trailed with an F1-measure of 0.33.

## 🛠 Models
- **SloBERTa**: A derivative of BERT model fine-tuned for Slovene language.
- **CroSloEngualBERT**: A BERT variant optimized for cross-lingual tasks encompassing Croatian, Slovenian, and English.

## 🗃 Dataset
The training ground comprises a publicly available labeled dataset of Twitter posts in English, paired with its Slovenian translation.

## 📊 Results
The evaluation revealed a significant disparity in model performance across diverse topics. The full-article analysis yielded the highest accuracy, with feminism leading the score chart with an F1-measure of 0.58 and atheism at the bottom with an F1-measure of 0.33.

## 🚀 Usage
Follow through for instructions on environment setup, model training, and evaluation.

How to reproduce: 

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target "Janez Jansa" --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target "Janez Jansa" --model_select EMBEDDIA/crosloengual-bert
```

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Atheism --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Atheism --model_select EMBEDDIA/crosloengual-bert
```

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Feminism --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Feminism --model_select EMBEDDIA/crosloengual-bert
```

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Climate --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Climate --model_select EMBEDDIA/crosloengual-bert
```

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Abortion --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Abortion --model_select EMBEDDIA/crosloengual-bert
```
