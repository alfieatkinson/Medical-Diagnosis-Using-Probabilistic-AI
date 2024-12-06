# Medical Diagnosis using Probabilistic AI: Bayesian Networks and Gaussian Processes for Predicting Dementia and Parkinson’s Disease

This assessment has been completed as part of the Advanced Artificial Intelligence module in partial fulfilment of the Degree of **Master of Science in Computer Science**.

## Assessment Overview

The objective of this assessment is to implement a software solution for medical diagnosis problems using probabilistic AI techniques. The task involves applying methods such as Discrete Bayesian Networks, Gaussian Bayesian Networks, and Gaussian Processes to analyse datasets related to dementia and Parkinson’s disease. Students are required to answer probabilistic queries based on the data and compare the performance of different AI methods in terms of predictive power, accuracy, and other relevant metrics. The solution must be implemented in Python, with a focus on model training, inference, and performance evaluation.

## Abstract

This project explores the application of probabilistic artificial intelligence models for medical diagnosis, specifically focusing on dementia and Parkinson’s disease prediction. We compare the performance of three models: Discrete Bayesian Networks (DBNs), Gaussian Bayesian Networks (GBNs), and Gaussian Processes (GPs), evaluating their ability to handle both discrete and continuous data. Performance metrics such as accuracy, F1-score, AUC, Brier score, and log loss are used to assess model effectiveness. Our findings indicate that DBNs provide superior performance in terms of accuracy and computational efficiency, while GBNs and GPs offer distinct advantages in handling continuous data. This study highlights the strengths, limitations, and computational costs of each model, offering recommendations for their application in medical diagnostics. For more detailed information, please refer to the [report.pdf](report.pdf) in this repository.

## Grade Achieved

- **Software**: 88/100
- **Report**: 95/100
- **Overall Mark**: 92/100

## Cloning and Requirements

1. Clone the repository:
    ```bash
    git clone https://github.com/alfieatkinson/Medical-Diagnosis-Using-Probabilistic-AI
    ```
2. Navigate to the project directory:
    ```bash
    cd Medical-Diagnosis-Using-Probabilistic-AI
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

This project uses two datasets for the medical diagnosis of dementia and Parkinson’s disease:

1. **Dementia Dataset**: The data used for dementia classification and prediction was sourced from the study by Battineni et al. (2019). The dataset is available online at:  
   [Machine learning in medicine: Classification and prediction of dementia by support vector machines (SVM)](https://doi.org/10.17632/tsy6rbc5d4.1)

2. **Parkinson’s Disease Dataset**: The dataset for Parkinson's disease, specifically focusing on dysphonia measurements for telemonitoring, is provided by Little et al. (2008) and can be accessed from Kaggle:  
   [Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set?select=parkinsons.data)

Both datasets contain relevant features for building machine learning models aimed at diagnosing dementia and Parkinson's disease.
