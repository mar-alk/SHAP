
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
conda install numpy pandas matplotlib seaborn scikit-learn shap tensorflow catboost rdkit
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
- `rdkit`

## Usage

Once the dependencies are installed, you can run the scripts and notebooks as follows:

1. **ECFP4 Analysis (Script 01)**:
   - Open the `ECFP4 Analysis.ipynb` notebook or run the corresponding script.
   - This script classifies molecules as Active or Inactive and predicts their pXC50 values using ECFP4 fingerprints.
   - SHAP analysis is applied to understand which molecular features most influence the prediction.

2. **ECFP6 Analysis (Script 02)**:
   - Run the `ECFP6 Analysis.ipynb` notebook or corresponding script.
   - Similar to the ECFP4 analysis, this script uses ECFP6 fingerprints for predicting activity and conducting SHAP feature importance analysis.

3. **Traceback Molecular Structures (Script 03)**:
   - Run the `Traceback Molecular Structures.ipynb` notebook or corresponding script.
   - This script maps SHAP-identified features back to their corresponding molecular substructures for an interpretable connection between features and molecular structures.

### Example: SHAP Analysis with CatBoost for Feature Importance

The following example demonstrates using SHAP with a trained CatBoost model to analyze feature importance and visualize the ECFP fingerprints of a specific molecule.

```python
# Import required libraries
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Generate predictions on the test dataset
all_preds = cat_model.predict(X_test)

# Convert test data to a DataFrame for SHAP analysis
X_df = pd.DataFrame(X_test)

# Deep copies of data for separate SHAP analysis and plotting
x_df = X_df.copy(deep=True)
x_df_1st = x_df.copy(deep=True)

# Store predictions in a separate DataFrame for first SHAP plot
x_df_1st['Predictions'] = all_preds

# Reset index for consistency in both DataFrames
x_df = x_df.reset_index(drop=True)
x_df_1st = x_df_1st.reset_index(drop=True)

# Apply SHAP for feature importance analysis using CatBoost model
shap_values = shap.TreeExplainer(cat_model).shap_values(x_df)

# Plot summary of SHAP feature importance
plt.figure(figsize=(10,10))
shap.summary_plot(shap_values, x_df, plot_size=(10,10), show=False, plot_type='dot', max_display=10)
plt.title('SHAP Feature Importance for CatBoost', weight='bold', size=20)
plt.xticks(size=20, weight='bold')
plt.yticks(size=20, weight='bold')
plt.savefig('Reg_SHAP_04.png', dpi=100, bbox_inches='tight')
plt.show()

# Calculate and display top 10 most important features
feature_imp = np.mean(np.abs(shap_values), axis=0)
ind = feature_imp.argsort()[-10:][::-1]  # Sort features by importance
print("Top 10 Important Features:", np.array(x_df.columns)[ind])
print("Feature Importance Scores:", feature_imp[ind])

### Traceback of Important ECFP Features to Molecular Substructure

# Select molecule from positive dataset
a = 1239  # Example index for molecule
smiles = Data_Positive.iloc[a, 0]

# Generate ECFP4 fingerprint and retrieve bit information
bitinfo = {}
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bitinfo, useFeatures=True)  # ECFP4 fingerprint

# Function to convert ECFP to DataFrame
def ecfp_to_dataframe(ecfp):
    arr = np.zeros((1, 1024))
    for i in ecfp.GetOnBits():
        arr[0, i] = 1
    df = pd.DataFrame(arr, columns=[f"Bit_{i}" for i in range(1024)])
    return df

# Convert ECFP fingerprint to DataFrame for inspection
df = ecfp_to_dataframe(fp)
print("ECFP4 Fingerprint DataFrame:\n", df)

# Find bits in the fingerprint that are set to 1 (on-bits)
on_bits = np.where(df.iloc[0, :] == 1)
print("On-bits in the fingerprint:", on_bits)

# Visualize molecular substructure corresponding to a specific ECFP bit
img = Draw.DrawMorganBit(mol, bit_id=4, bitInfo=bitinfo)  # Example bit ID: 4
img.show()

------------------------------------------------------------------
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
