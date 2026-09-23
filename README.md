# [Attention Spiking Neural Networks](https://ieeexplore.ieee.org/document/10032591)

# [Attention Spiking Neural Networks - Supplementary Materials](https://github.com/BICLab/Attention-SNN/issues/3)

# DVS Gesture — Downstream Classification

This repository contains the implementation for the **DVS Gesture downstream classification task** using the Attention-SNN framework.

The implementation is provided as a Google Colab workflow and uses the `task-downstream-dvsGesture` branch of the repository.

## Overview

The workflow consists of the following stages:

```text
DVS Gesture Dataset
        │
        ▼
Download / Prepare Dataset
        │
        ▼
DVS_Gesture.py
        │
        ▼
Processed DVS Gesture Data
        │
        ▼
Attention-SNN + CNN Classification
        │
        ▼
Classification Results
```

## Repository

The implementation is hosted in:

**GitHub:** LovelyDev-06/Attention-SNN1

The notebook uses the following branch:

```text
task-downstream-dvsGesture
```

## Dataset

This implementation uses the **DVS Gesture dataset**.

The dataset archive is expected to be available in Google Drive as:

```text
DvsGesture.tar.gz
```

The notebook copies this archive into:

```text
Attention-SNN1/MA_SNN/DVSGestures/data/
```

## Requirements

The implementation is designed to run in **Google Colab**.

The notebook uses:

* Python
* Google Colab
* Google Drive
* PyTorch-based SNN implementation
* DVS Gesture event-based dataset
* Attention-SNN/CNN classification implementation

The required project dependencies are contained within the cloned repository.

---

## Implementation

### 1. Clone the repository

The required branch is cloned directly from GitHub:

```bash
!git clone -b task-downstream-dvsGesture https://github.com/LovelyDev-06/Attention-SNN1.git
```

This downloads the downstream DVS Gesture implementation.

---

### 2. Navigate to the dataset directory

Move into the directory containing the DVS Gesture data-processing code:

```python
%cd Attention-SNN1/MA_SNN/DVSGestures/data
```

---

### 3. Mount Google Drive

The notebook mounts Google Drive so that the DVS Gesture dataset archive can be accessed:

```python
from google.colab import drive

drive.mount('/content/drive')
```

---

### 4. Copy the DVS Gesture dataset

The dataset archive is copied from Google Drive into the project's data directory:

```bash
!cp "/content/drive/MyDrive/DvsGesture.tar.gz" \
"/content/Attention-SNN1/MA_SNN/DVSGestures/data/"
```

Therefore, before running the notebook, make sure the following file exists in Google Drive:

```text
MyDrive/
└── DvsGesture.tar.gz
```

---

### 5. Prepare the DVS Gesture data

The dataset preparation step is executed using:

```bash
!python DVS_Gesture.py
```

`DVS_Gesture.py` is the preprocessing/data preparation component used by this implementation.

This step should be completed before starting the classification model.

---

### 6. Return to the main project directory

After preparing the dataset, the notebook moves one directory level up:

```python
%cd ..
```

---

### 7. Run downstream classification

The downstream classification implementation is executed using the Python module:

```bash
!python -m DVSGestures.Att_SNN_CNN
```

The classification implementation is contained in:

```text
DVSGestures/
└── Att_SNN_CNN
```

This runs the Attention-SNN/CNN-based downstream classification experiment on the prepared DVS Gesture data.

---

## Complete Colab Workflow

The complete implementation can be executed in the following order:

```bash
# Clone the required branch
!git clone -b task-downstream-dvsGesture https://github.com/LovelyDev-06/Attention-SNN1.git

# Enter dataset directory
%cd Attention-SNN1/MA_SNN/DVSGestures/data

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset
!cp "/content/drive/MyDrive/DvsGesture.tar.gz" \
"/content/Attention-SNN1/MA_SNN/DVSGestures/data/"

# Prepare the dataset
!python DVS_Gesture.py

# Return to project directory
%cd ..

# Run downstream classification
!python -m DVSGestures.Att_SNN_CNN
```

## Project Structure

The relevant implementation structure used by the notebook is:

```text
Attention-SNN1/
│
└── MA_SNN/
    │
    └── DVSGestures/
        │
        ├── data/
        │   ├── DvsGesture.tar.gz
        │   └── DVS_Gesture.py
        │
        └── Att_SNN_CNN
```

The exact additional files and directories are maintained by the repository branch.

## Reproducing the Experiment

To reproduce the implementation:

1. Open the notebook in Google Colab.
2. Clone the `task-downstream-dvsGesture` branch.
3. Mount Google Drive.
4. Place `DvsGesture.tar.gz` in the root of Google Drive.
5. Copy the dataset into the project's `DVSGestures/data` directory.
6. Run `DVS_Gesture.py` to prepare the dataset.
7. Run `DVSGestures.Att_SNN_CNN` for downstream classification.

## Colab

The notebook provides a direct **Open in Colab** link and can be used as the entry point for reproducing the experiment.

## Implementation Summary

The implementation follows this pipeline:

```text
GitHub Repository
       │
       ▼
task-downstream-dvsGesture branch
       │
       ▼
DVS Gesture Dataset
       │
       ▼
DVS_Gesture.py
       │
       ▼
Prepared Dataset
       │
       ▼
Att_SNN_CNN
       │
       ▼
DVS Gesture Downstream Classification
```

## Notes

* The implementation is configured for Google Colab.
* The DVS Gesture dataset archive is loaded from Google Drive.
* The dataset preparation is handled by `DVS_Gesture.py`.
* The downstream classification experiment is launched through `DVSGestures.Att_SNN_CNN`.
* The notebook itself contains the complete execution sequence required for the experiment.


### 5. Extra

1. The implementation of Att-VGG-SNN in https://github.com/ridgerchu/SNN_Attention_VGG

2. /module/Attention.py defines the  Attention layer and /module/LIF.py,LIF_Module.py defines LIF module.

3. The CSA-MS-ResNet104 model is available at https://pan.baidu.com/s/1Uro7IVSerV23OKbG8Qn6pQ?pwd=54tl (Code: 54tl).

   

## **Citation**
```
@ARTICLE{10032591,
  author={Yao, Man and Zhao, Guangshe and Zhang, Hengyu and Hu, Yifan and Deng, Lei and Tian, Yonghong and Xu, Bo and Li, Guoqi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention Spiking Neural Networks}, 
  year={2023},
  volume={45},
  number={8},
  pages={9393-9410},
  doi={10.1109/TPAMI.2023.3241201}}
```
