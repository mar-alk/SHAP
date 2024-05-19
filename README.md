Molecular Activity Prediction and Feature Analysis
This repository contains scripts for analyzing molecular fingerprints and predicting molecular activity using machine learning algorithms. The analysis involves the use of ECFP04 and ECFP06 fingerprints, and SHAP analysis for feature importance. The workflow is as follows:

ECFP04 Analysis (Script 01):
Objective: Classify molecules as Active or Inactive and predict pxC50 values.
Method: Use multiple machine learning algorithms to determine the best-performing model.
Feature Analysis: Apply SHAP to identify features contributing to molecular activity.

ECFP06 Analysis (Script 02):
Objective: Classify active molecules and predict pxC50 values.
Method: Similar approach as ECFP04, using multiple machine learning algorithms.
Feature Analysis: Apply SHAP to identify features contributing to increased molecular activity.

Traceback Molecular Structures (Script 03):
Objective: Identify which features correspond to specific molecules.
Method: Traceback analysis to connect SHAP-identified features to original molecular structures.

Files
ECFP04 Analysis: Script for ECFP04 fingerprint analysis.
ECFP06 Analysis: Script for ECFP06 fingerprint analysis.
Traceback Molecular Structures: Script for tracing back main features (ECFP Bits) to identify corresponding substructure.
