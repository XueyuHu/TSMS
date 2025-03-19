# TSMS
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)

TSMS is an AI-driven energy materials discovery framework that integrates high-throughput computations (HTCs)<sup>1</sup>, standardized experiments, and active learning.

- [Overview](#Overview)
- [SystemRequirements](#SystemRequirements)
- [InstallationGuide](#InstallationGuide)
- [Demo](#Demo)
- [Instructions for Use](#InstructionsforUse)
- [Publication](#Publication)
- [Reference](#Reference)

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



# Demo
**Training**

<code>python ML.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull -parm False </code>

**Cross-Validation**

<code>python.exe ML.py -model xgb -stage 1 -train Data/dataset_1.csv -targ Ehull </code>

**Prediction**

<code>python ML.py -model xgb -stage 1 -train Data/train_1.csv -shap True -test Data/test_1.csv -targ Ehull -pth pred_ </code>

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
