# TSMS
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)

TSMS is an AI-driven energy materials discovery framework that integrates high-throughput computations (HTCs)<sup>1</sup>, standardized experiments, and active learning.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-Guide)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)
- [Publication](#publication)
- [Reference](#reference)

# Overview
<img src="https://github.com/user-attachments/assets/818488df-3296-4768-be34-ec5d20106352" width="60%">

**T**wo-**S**tage **M**aterial **S**creening (TSMS) follows a hierarchical screening approach as shown in the above Figure: First, HTCs densely sample computationally derived functional information to delineate potential regions of interest within an uncharted chemical space. Then, standardized experiments provide discrete sampling of promising candidates, iteratively guided by an active learning framework until the AI model achieves optimal accuracy. Finally, the AI-driven model then constructs a high-resolution topographic mapping of chemical space. Simultaneously, feature attribution methods identify the decisive factors governing high-dimensional properties, providing critical insights for rational material design.


# SystemRequirements
## Hardware Requirements
This software runs on any standard computer with a modern operating system.

## Software Requirements
### OS Requirements
- Windows 10/11
- Linux 9.4

### Python Dependencies
<code>TSMS</code> mainly depends on the Python scientific stack.
```python
sys
json
pandas
numpy
tqdm
dataset
warnings
xgboost
joblib
sklearn
shap
argparse
matplotlib
```

# InstallationGuide
<code>TSMS</code> does not require additional intallation as long as the Python dependencies are properly set up.


# Demo
## Instructions
### Three Critical Files
<code>/TSMS/Data/dataset_1_20250319.csv</code> contains the latest dataset for training in the First Stage Machine Learning. 

<code>/TSMS/Data/dataset_2_20250319.csv</code> contains the latest dataset for training in the Second Stage Active Learning.

<code>/TSMS/ML.py</code> implements the machine learning process for all training using the <code>XGBoost</code> algorithm.

### Three Critical Procedures
**Training**
```python
python ML.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull -parm False
```
**Cross-Validation**
```python
python ML.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull
```
**Prediction**
```python
python ML.py -model xgb -stage 1 -train Data/train_1.csv -shap True -test Data/test_1.csv -targ Ehull -pth pred_
```

### Optional tags
```python
    parser.add_argument('-model', default='xgb', type=str, help='model')
    parser.add_argument('-stage', default=1, type=int, help='stage')
    parser.add_argument('-num_split', default=5, type=int, help='num split')
    parser.add_argument('-seed', default=2023, type=int, help='random seed')
    parser.add_argument('-train', default='Data/train_1.csv', type=str, help='input path')
    parser.add_argument('-targ', default=1, help='target either int or str')
    parser.add_argument('-test', default='Data/test_1.csv', help='Data/test_1.csv')
    parser.add_argument('-pred', default=False, help='Normalize or not')
    parser.add_argument('-drop', default=False, help='Drop or not')
    parser.add_argument('-shap', default=False, help='Shap value analysis')
    parser.add_argument('-pth', default=False, help='Checkpoint output path')

    parser.add_argument('-depth', default=False, help='max_depth')
    parser.add_argument('-leaves', default=False, help='max_leaves')
    parser.add_argument('-child', default=False, help='min child weight')
    parser.add_argument('-lr', default=False, help='learning rate')
    parser.add_argument('-n', default=False, help='n estimators')
    parser.add_argument('-parm', default=False)
```

## Expected output
**Training:**
The optimal set of hyperparamters will be determined and provided.

**Cross-Validation:**
Cross-validation will be performed by varying the random seed, and both the best and averaged performance metrics will be reported.

**Prediction:**
The predicted values will be saved in <code>/TSMS/Result/</code>, along with the corresponding ground truth and candidate information.


## Expected run time for demo on a "normal" desktop computer
The demo has been evaluated running on a single Intel(R) CPU i7-11700 (2.5GHz) core laptop.

**Training:**
Several hours

**Cross-Validation:**
Varies depending on the number of random seed selections

**Prediction:**
Less than 0.01s per candidate.

# Instructions for Use

# Publication

```bibtex
@article{Hu_2025_TSMS,
  title={Mechanistically Interpretable AI for Accelerated Energy Materials Design},
  DOI={},
  journal={},
  author={Xueyu Hu, Ke Liao, Yucun Zhou, Haoyu Li, Zheyu Luo, Nai Shi, Yong Ding, Weining Wang, Weilin Zhang, Doyeub Kim, Chanho Kim, Yoojin Ahn, Nikhil Govindarajan, Zhijun Liu*, and Meilin Liu*},
  year={},
  pages={}
}
```


# Reference
1. http:/doi.org/10.1039/D4EE03762F
