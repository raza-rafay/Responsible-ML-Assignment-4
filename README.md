Assignment 4 – Robustness, Generalization, and Dataset Drift Audit of COMPAS Models

Name: Rafay Raza
GWID: G40856805

Purpose of the Analysis

The purpose of this analysis is to evaluate the reliability of machine learning models trained on the COMPAS two-year recidivism dataset beyond standard accuracy metrics. The audit focuses on three key areas: generalization, distribution drift, and robustness.

The analysis compares a logistic regression model and a gradient-boosted tree model to assess whether they learn stable patterns or rely on spurious correlations. Distribution drift is evaluated using Population Stability Index (PSI), Kolmogorov–Smirnov (KS) tests, and Maximum Mean Discrepancy (MMD) to determine whether the data generating process has changed.

Robustness is assessed through stress testing techniques including slice-based evaluation, counterfactual swaps, and sensitivity analysis. Model behavior is further examined using LIME and SHAP explanations, and counterfactual examples are generated using DiCE. The goal is to determine whether the models remain reliable under changing conditions and across subgroups.

Python Libraries Used

The following Python libraries were used in this assignment:

pandas
numpy
matplotlib
scikit-learn
statsmodels
scipy
lime
shap
dice-ml
solas-ai

Instructions for Reproducing the Results

To reproduce the results:

Install the required Python libraries:
pip install pandas numpy matplotlib scikit-learn statsmodels scipy lime shap dice-ml solas-ai

Open the Jupyter Notebook file:
RafayRaza_G40856805_Assignment4.ipynb

Run all cells in the notebook from top to bottom. The notebook will:

Load and clean the COMPAS dataset
Perform exploratory analysis and reproduce baseline COMPAS findings
Train logistic regression and gradient-boosted models
Evaluate performance metrics (accuracy, AUC, log loss, FPR, FNR) overall and by race
Conduct fairness analysis using AIR, ME, and SMD
Perform intersectional subgroup analysis (race × gender)
Generate model explanations using LIME and SHAP
Produce counterfactual explanations using DiCE
Evaluate generalization through train vs test comparisons
Measure distribution drift using PSI, KS test, and MMD
Run robustness checks including slice-based evaluation and sensitivity analysis

The dataset is loaded directly from the ProPublica COMPAS dataset repository:
https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

AI Acknowledgment

AI tools were used for general programming guidance, debugging, and assistance with Markdown formatting and grammar.
