import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

def discrete_cross_validation(df: pd.DataFrame, nsplits: int = 5, structure_kwargs: dict = None, parameter_kwargs: dict = None) -> dict[str, any]:
    """
    Perform k-fold cross-validation on a Bayesian Network using structure and parameter learning functions.
    
    Parameters:
    - df: The dataframe containing the data for the Bayesian Network.
    - nsplits: The number of splits for cross-validation (default is 5).
    - structure_kwargs: Additional arguments for the structure learning function (dict).
    - parameter_kwargs: Additional arguments for the parameter learning function (dict).
    
    Returns:
    - A dictionary of average performance metrics (e.g., accuracy, precision, recall, F1 score).
    - True labels and predicted labels for all folds, which can be used for confusion matrix.
    """