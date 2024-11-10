import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

def discrete_cross_validation(df: pd.DataFrame, nsplits: int = 5, structure_kwargs: dict = None, parameter_kwargs: dict = None) -> tuple:
    """
    Cross-validate a model by splitting data into training and test sets,
    applying structure learning and then parameter learning with bnlearn.
    
    Parameter:
        df: The dataset to be used for cross-validation (pandas DataFrame).
        nsplits: The number of splits for K-fold cross-validation (default 5).
        structure_kwargs: Keyword arguments for the structure learning function.
        parameter_kwargs: Keyword arguments for the parameter learning function.
        
    Returns:
        tuple: 
            - The trained model after the last fold.
            - A dictionary of average metrics across all folds.
    """