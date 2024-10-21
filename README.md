
# Molecular Activity Prediction and Feature Analysis

This repository contains scripts for predicting molecular activity using machine learning algorithms and analyzing the importance of molecular features. The analysis involves two types of molecular fingerprints, **ECFP4** and **ECFP6**, and uses **SHAP** (SHapley Additive exPlanations) for feature importance analysis. The workflow is structured as follows:

## Folder Structure

### 1. **Dataset Folder**
Contains datasets with ECFP4 and ECFP6 molecular fingerprint representations for training and testing machine learning models.

### 2. **Implementation Folder**
Contains the following scripts for molecular activity prediction and feature analysis:

#### **ECFP4 Analysis (Script 01):**
- **Objective**: Classify molecules as Active or Inactive, and predict their pXC50 values.
- **Method**: Multiple machine learning algorithms are employed to evaluate and determine the best-performing model for molecular activity prediction.
- **Feature Analysis**: SHAP analysis is applied to identify which molecular features (fingerprint bits) contribute the most to molecular activity.

#### **ECFP6 Analysis (Script 02):**
- **Objective**: Classify active molecules and predict their pXC50 values.
- **Method**: A similar approach to **ECFP4 Analysis**, using multiple machine learning models to predict molecular activity and evaluate their performance.
- **Feature Analysis**: SHAP analysis is applied to determine the features responsible for increased molecular activity in ECFP6 representations.

#### **Traceback Molecular Structures (Script 03):**
- **Objective**: Identify the specific molecular substructures that correspond to the features identified by SHAP as most important.
- **Method**: Traceback analysis is used to map SHAP-identified important features (ECFP bits) back to the original molecular structures, allowing for a direct connection between features and molecular substructures.

## Files

- **ECFP4 Analysis**: Script for analyzing molecular fingerprints using ECFP4 representation, predicting activity, and performing SHAP-based feature analysis.
- **ECFP6 Analysis**: Script for analyzing molecular fingerprints using ECFP6 representation, predicting activity, and performing SHAP-based feature analysis.
- **Traceback Molecular Structures**: Script for tracing back the most influential features identified by SHAP to their corresponding molecular substructures, providing interpretable insights into the specific parts of a molecule driving biological activity.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/mar-alk/SHAP.git
cd SHAP
```

### Using pip:
Install the required dependencies:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, you can manually install the required libraries (see [Dependencies](#dependencies)).

### Using conda:
Alternatively, you can use conda to create a new environment and install dependencies:

```bash
conda create --name shap_qsar python=3.8
conda activate shap_qsar
conda install numpy pandas matplotlib seaborn scikit-learn shap tensorflow catboost
```

## Dependencies

The following Python libraries are required for running the notebooks and scripts:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `shap`
- `tensorflow`
- `catboost`

## Usage

Once the dependencies are installed, you can run the scripts and notebooks as follows:

1. **ECFP4 Analysis (Script 01)**:
   - Run the `ECFP4 Analysis.ipynb` notebook or corresponding script.
   - It will classify molecules as Active or Inactive and predict pXC50 values based on ECFP4 fingerprints.
   - SHAP analysis will provide insights into which molecular features contribute the most to the predicted activity.

2. **ECFP6 Analysis (Script 02)**:
   - Run the `ECFP6 Analysis.ipynb` notebook or corresponding script.
   - Similar to the ECFP4 analysis, but using ECFP6 fingerprints for predicting activity and conducting SHAP feature importance analysis.

3. **Traceback Molecular Structures (Script 03)**:
   - Run the `Traceback Molecular Structures.ipynb` notebook or corresponding script.
   - This script will map SHAP-identified important features back to their corresponding molecular substructures, providing interpretable connections between features and molecular structures.

## SHAP Analysis for Feature Importance

SHAP is utilized in both ECFP4 and ECFP6 analyses to provide interpretability to the model predictions. SHAP values explain the importance of each feature (molecular fingerprint bit) in driving the model's prediction, allowing for a detailed understanding of how molecular structures affect biological activity.

The **Traceback Molecular Structures** script further enhances this by mapping important features identified by SHAP back to the corresponding molecular substructures, offering insight into which specific parts of a molecule contribute to its activity.
