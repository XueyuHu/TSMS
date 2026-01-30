# TSMS
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)

TSMS is an AI-driven energy materials discovery framework that integrates high-throughput computations (HTCs)<sup>1</sup>, standardized experiments, and active learning.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)
- [Publication](#publication)
- [Reference](#reference)

# Overview
![New Microsoft PowerPoint Presentation1](https://github.com/user-attachments/assets/9928f7ec-18be-49f7-8255-2c2ba1ea8d07)


**T**wo-**S**tage **M**aterial **S**creening (TSMS) follows a hierarchical screening approach as shown in the above Figure: First, HTCs densely sample computationally derived functional information to delineate potential regions of interest within an uncharted chemical space. Then, standardized experiments provide discrete sampling of promising candidates, iteratively guided by an active learning framework until the AI model achieves optimal accuracy. Finally, the AI-driven model then constructs a high-resolution topographic mapping of chemical space. Simultaneously, feature attribution methods identify the decisive factors governing high-dimensional properties, providing critical insights for rational material design.

# System Requirements
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

# Installation Guide
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
python ML_v2.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull
# Open fold-safe scaling
python ML_v2.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull -pred
# GroupKFold
python ML_v2.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull -pred -group_cv
```

**Cross-Validation**
```python
python ML_v2.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull -pred -group_cv -num_split 5 -seed 2023 -n_jobs 1
```
**Prediction**
```python
python ML_v2.py -model xgb -stage 1 -train Data/train_1.csv -test Data/test_1.csv -targ Ehull -pred -drop -shap
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
## How to run the software on your data
<img src="https://github.com/user-attachments/assets/9e0ff4da-ad9d-46c8-acbd-22dc95272233" width="60%">

For a given candidate, the process follows the **Flowchart procedure** to generate all relevant properties for its role as an oxygen electrode in PCECs.  

The first step is **Material Embedding**, which extracts its **Fundamental Features**, including **Chemical Composition**, **Structural Properties**, and **Physicochemical Properties**. Next, **First-Stage Machine Learning** is applied to derive its **Functional Information**, consisting of **Computational Parameters & Descriptors**. These **Fundamental Features** and **Functional Information** together serve as inputs for the **Second-Stage Active Learning**.  

Through multiple iterations guided by active learning, **TSMS** can achieve highly accurate predictions of **Experimental Performance**. Subsequently, three post-evaluation processes are conducted: **Ground Truth Comparison** to assess **Accuracy**, **SHAP Analysis** to enhance **Interpretability**, and an **Out-of-Distribution Task** to validate **Generalizability**.  

This establishes the **AI-driven Two-Stage Material Screening** methodology.

## Reproduction instructions

Several points need clarification: 

**1. Physicochemical Properties**

To describe the constituent elements of each candidate, we enumerated eleven fundamental physicochemical properties: ionic radius, electron affinity, oxidation state, melting point, boiling point, atomic mass, elemental density, ionization energy, electronegativity, calcination index (melting point of the corresponding oxide), and chemical potential of the oxide. These properties were used to derive 132 distinct descriptors through twelve linear combinations designed for the cubic perovskite crystal structure: 

(1) Weighted average of the property for A-site elements, based on their fractions.

(2) Weighted average of the property for B-site elements, based on their fractions.

(3) Weighted average of the property for A-site dopant elements, based on their fractions.

(4) Weighted average of the property for B-site dopant elements, based on their fractions.

(5) Weighted average of the property for A-site major elements, based on their fractions.

(6) Weighted average of the property for B-site active elements, based on their fractions.

(7) Sum of (5) and (6).

(8) Ratio of (1) to (2).

(9) Ratio of (5) to (6).

(10) Ratio of (3) to (1).

(11) Ratio of (4) to (2).

(12) Sum of (3) and (4) divided by the sum of (5) and (6).

Physicochemical properties are uniformly represented in the form Property_#, where “Property” denotes the specific physicochemical property and “#” refers to the linear combination method.

**2. Synthesizability**

Synthesizability was assessed using the code developed by Gu et al.<sup>2</sup>

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
1. https://pubs.rsc.org/en/content/articlehtml/2024/ee/d4ee03762f
2. https://www.nature.com/articles/s41524-022-00757-z
