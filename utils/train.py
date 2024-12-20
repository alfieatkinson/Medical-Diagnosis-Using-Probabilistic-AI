import warnings
import logging
import pandas as pd
import numpy as np
import bnlearn as bn
from pgmpy.global_vars import logger
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, log_loss
)
from pgmpy.estimators import HillClimbSearch, PC, BDeuScore
from pgmpy.models import BayesianModel, LinearGaussianBayesianNetwork
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.stats import entropy
import time

warnings.filterwarnings('ignore')
logger.setLevel(logging.ERROR)
logging.getLogger('seaborn').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

def discrete_cross_validation(df: pd.DataFrame, target: str, nsplits: int = 5, structure_kwargs: dict = None, parameter_kwargs: dict = None) -> dict[str, any]:
    """
    Perform k-fold cross-validation using Bayesian Networks for discrete data.
    
    Parameters:
        df (DataFrame): The dataframe containing the data.
        target (str): The target column name.
        nsplits (int): The number of splits for cross-validation. Defaults to 5.
    
    Returns:
        dict: A dictionary containing:
            - metrics (dict): A dictionary of average performance metrics:
                - classification (dict): Classification metrics:
                    - accuracy (float)
                    - precision (float)
                    - recall (float)
                    - f1_score (float)
                - roc_auc (float)
                - brier_score (float)
                - log_loss (float)
            - confusion_matrix (array): Confusion matrix of all folds.
            - time_taken (float): Time taken for the entire cross-validation process in seconds.
    """
    start_time = time.time()  # Track the start time of the cross-validation
    
    # Setup k-fold cross-validation
    kf = KFold(n_splits=nsplits)
    
    # Initialise lists to hold metric results
    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    roc_auc_results = []
    brier_score_results = []
    log_loss_results = []

    all_y_true = []
    all_y_pred = []
    
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{nsplits}")
        
        # Split data into training and test sets
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        
        model = None
        learned_model = None
        
        # Structure learning
        if structure_kwargs is None:
            structure_kwargs = {}
        model = bn.structure_learning.fit(train_data, **structure_kwargs)
        
        # Parameter learning
        if parameter_kwargs is None:
            parameter_kwargs = {}
        learned_model = bn.parameter_learning.fit(model, train_data, **parameter_kwargs)
        
        # Drop columns from test data not in the model nodes
        model_nodes = model['model'].nodes()
        test_columns = test_data.columns.tolist()
        columns_to_drop = [col for col in test_columns if col not in model_nodes]
        test_data_filtered = test_data.drop(columns=columns_to_drop)
        
        # Reset lists for true and predicted values at the start of each fold
        fold_y_true = []
        fold_y_pred = []
        
        # Loop over each row in the test set and predict individually
        for idx, row in test_data_filtered.iterrows():
            # Prepare individual row as DataFrame
            row_df = row.to_frame().T
            
            # Get true value for the row
            y_true = row_df[target].values[0]
            
            try:
                # Make prediction for this row
                y_pred = bn.predict(learned_model, row_df, target)[target].values[0]
                
                # Collect true and predicted values for confusion matrix
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
            
            except KeyError as e:
                # Handle specific KeyError related to state name not found
                print(f"Error for row {idx} in fold {fold+1}: KeyError - {e}")
                print(f"Row causing error: {row_df}")
                print(f"Model state names for target variable: {learned_model['model'].get_cpds(target).state_names}")
                print(f"Model nodes: {model_nodes}")
                continue  # Skip this row and move to the next
            
            except Exception as e:
                # Catch all other exceptions and log them
                print(f"Unexpected error for row {idx} in fold {fold+1}: {e}")
                print(f"Row causing error: {row_df}")
                continue  # Skip this row and move to the next
        
        # Compute evaluation metrics for this fold
        accuracy_results.append(accuracy_score(all_y_true, all_y_pred))
        precision_results.append(precision_score(all_y_true, all_y_pred, average='binary', zero_division=1))
        recall_results.append(recall_score(all_y_true, all_y_pred, average='binary', zero_division=1))
        f1_results.append(f1_score(all_y_true, all_y_pred, average='binary'))
        
        try:
            roc_auc_results.append(roc_auc_score(all_y_true, all_y_pred))
        except ValueError as e:
            print(f"Cannot calculate auc score for this fold: {e}")
        
        brier_score_results.append(brier_score_loss(all_y_true, all_y_pred))
        try:
            log_loss_results.append(log_loss(all_y_true, all_y_pred))
        except ValueError as e:
            print(f"Cannot calculate log loss for this fold: {e}")

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        # Compute average metrics across folds
        metrics = {
            'classification': {
                'accuracy': np.mean(accuracy_results),
                'precision': np.mean(precision_results),
                'recall': np.mean(recall_results),
                'f1_score': np.mean(f1_results),
            },
            'roc_auc': np.mean(roc_auc_results) if roc_auc_results else np.nan,
            'brier_score': np.mean(brier_score_results),
            'log_loss': np.mean(log_loss_results) if log_loss_results else np.nan
        }
        
        # Compute confusion matrix
        confusion = confusion_matrix(all_y_true, all_y_pred)
        
        # Calculate the time taken for the whole cross-validation process
        time_taken = time.time() - start_time
        
        # Return metrics, confusion matrix, and time taken
        return {
            'metrics': metrics,
            'confusion_matrix': confusion,
            'time_taken': time_taken
        }
    else:
        print("No valid folds were processed.")
        return None

def gaussian_cross_validation(df: pd.DataFrame, target: str, nsplits: int = 5, structure_kwargs: dict = None) -> dict[str, any]:
    """
    Perform k-fold cross-validation on a Gaussian Bayesian Network using structure and parameter learning functions, querying each row individually for each fold. It continues even if a row causes an error and prints debug messages.
    
    Parameters:
        df (DataFrame): The dataframe containing the data for the Bayesian Network.
        target (str): The target column name.
        nsplits (int): The number of splits for cross-validation. Defaults to 5.
        structure_kwargs (dict): Additional arguments for the structure learning function, including 'methodtype' (either 'hc' for Hill Climbing or 'pc' for PC algorithm).
    
    Returns:
        A dictionary of average performance metrics (e.g., accuracy, precision, recall, F1 score, ROC AUC, Brier score, Log Loss),
        True labels and predicted labels for all folds, which can be used for confusion matrix.
    """
    if structure_kwargs is None:
        structure_kwargs = {}

    kf = KFold(n_splits=nsplits)

    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    roc_auc_results = []
    brier_score_results = []
    log_loss_results = []
    
    all_y_true = []  # Accumulate true labels across all folds
    all_y_pred = []  # Accumulate predicted labels across all folds

    # Start timer for cross-validation
    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{nsplits}")

        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        # Choose structure learning method
        if structure_kwargs.get('methodtype', 'hc') == 'hc':
            print("Using Hill Climbing with BDEU Score for structure learning.")
            hc_search = HillClimbSearch(train_data)
            best_model = hc_search.estimate(scoring_method=BDeuScore(train_data))
        elif structure_kwargs.get('methodtype', 'hc') == 'pc':
            print("Using PC Algorithm for structure learning.")
            pc_search = PC(train_data)
            best_model = pc_search.estimate()
        else:
            raise ValueError("Invalid methodtype in structure_kwargs. Choose 'hc' or 'pc'.")

        # Check if target is in the learned model's edges
        if target not in [edge[0] for edge in best_model.edges()] and target not in [edge[1] for edge in best_model.edges()]:
            print(f"Skipping fold {fold+1} as {target} is not in the learned model.")
            continue  # Skip this fold if target is not in the model

        # Using Linear Gaussian Bayesian Network for continuous data
        learned_model = LinearGaussianBayesianNetwork(best_model.edges()) 
        learned_model.fit(train_data)

        # Predict using the learned model
        df_predict = test_data.drop([target], axis=1)  # Drop the target column for prediction

        # Get the predicted means and variances for all variables
        variable_order, means, var = learned_model.predict(df_predict)

        # The 'means' variable contains the predicted values for all target variables
        pred = means[:, variable_order.index(target)]  # Extract predicted values for target
        # Apply sigmoid to map continuous predictions to [0, 1]
        pred = 1 / (1 + np.exp(-pred))

        # Threshold the continuous predicted values to binary classification
        pred_binary = [1 if p > 0.5 else 0 for p in pred]  # Apply a 0.5 threshold to convert to binary

        # Append true and predicted values for metrics computation
        all_y_true.extend(test_data[target].values)
        all_y_pred.extend(pred_binary)

        # Evaluation metrics for this fold
        accuracy_results.append(accuracy_score(test_data[target], pred_binary))
        precision_results.append(precision_score(test_data[target], pred_binary, average='binary', zero_division=1))
        recall_results.append(recall_score(test_data[target], pred_binary, average='binary', zero_division=1))
        f1_results.append(f1_score(test_data[target], pred_binary, average='binary'))
        try:
            roc_auc_results.append(roc_auc_score(test_data[target], pred_binary))
        except ValueError as e:
            print(f"Cannot calculate auc score for this fold: {e}")
        brier_score_results.append(brier_score_loss(test_data[target], pred))
        try:
            log_loss_results.append(log_loss(test_data[target], pred))
        except ValueError as e:
            print(f"Cannot calculate log loss for this fold: {e}")

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        # Compute average metrics across folds
        metrics = {
            'classification': {
                'accuracy': np.mean(accuracy_results),
                'precision': np.mean(precision_results),
                'recall': np.mean(recall_results),
                'f1_score': np.mean(f1_results),
            },
            'roc_auc': np.mean(roc_auc_results) if roc_auc_results else np.nan,
            'brier_score': np.mean(brier_score_results),
            'log_loss': np.mean(log_loss_results) if log_loss_results else np.nan
        }
        
        # Compute confusion matrix
        confusion = confusion_matrix(all_y_true, all_y_pred)
        
        # Calculate the time taken for the whole cross-validation process
        time_taken = time.time() - start_time
        
        # Return metrics, confusion matrix, and time taken
        return {
            'metrics': metrics,
            'confusion_matrix': confusion,
            'time_taken': time_taken
        }
    else:
        print("No valid folds were processed.")
        return None

def gaussian_process_cross_validation(df: pd.DataFrame, target: str, kernel_type: str = 'rbf', nsplits: int = 5) -> dict[str, any]:
    """
    Perform k-fold cross-validation on a dataset using Gaussian Process regression for continuous evidence, 
    converting the predictions to binary classification for a binary target.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        target (str): The target column name (binary classification).
        kernel_type (str): The kernel type to use ('rbf' for RBF kernel, 'matern' for Matern kernel). Defaults to 'rbf'.
        nsplits (int): The number of splits for cross-validation. Defaults to 5.

    Returns:
        A dictionary containing average performance metrics (accuracy, precision, recall, f1_score, ROC AUC, Brier score, Log Loss),
        True labels and predicted labels for all folds, which can be used for confusion matrix.
    """
    # Setup k-fold cross-validation
    kf = KFold(n_splits=nsplits)
    
    # Lists to store performance metrics
    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    roc_auc_results = []
    brier_score_results = []
    log_loss_results = []

    all_y_true = []
    all_y_pred = []

    # Define the kernels
    if kernel_type == 'rbf':
        kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))  # RBF kernel
    elif kernel_type == 'matern':
        kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=1.5)  # Matern kernel
    else:
        raise ValueError("Invalid kernel_type. Choose 'rbf' or 'matern'.")

    # Start timer for cross-validation
    start_time = time.time()

    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{nsplits}")

        # Split data into training and test sets
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        # Separate features and target
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        X_test = test_data.drop(columns=[target])
        y_test = test_data[target]
        
        # Initialize and fit Gaussian Process Regressor
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        gp.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred_continuous, _ = gp.predict(X_test, return_std=True)
        
        # Apply sigmoid to map continuous predictions to [0, 1]
        y_pred_continuous = 1 / (1 + np.exp(-y_pred_continuous))

        # Convert continuous predictions to binary using a threshold of 0.5
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred_continuous]
        
        # Append true and predicted values for metrics computation
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred_binary)

        # Compute evaluation metrics for this fold
        accuracy_results.append(accuracy_score(y_test, y_pred_binary))
        precision_results.append(precision_score(y_test, y_pred_binary, average='binary'))
        recall_results.append(recall_score(y_test, y_pred_binary, average='binary'))
        f1_results.append(f1_score(y_test, y_pred_binary, average='binary'))
        try:
            roc_auc_results.append(roc_auc_score(all_y_true, all_y_pred))
        except ValueError as e:
            print(f"Cannot calculate auc score for this fold: {e}")
        brier_score_results.append(brier_score_loss(y_test, y_pred_continuous))
        try:
            log_loss_results.append(log_loss(y_test, y_pred_continuous))
        except ValueError as e:
            print(f"Cannot calculate log loss for this fold: {e}")

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        # Compute average metrics across folds
        metrics = {
            'classification': {
                'accuracy': np.mean(accuracy_results),
                'precision': np.mean(precision_results),
                'recall': np.mean(recall_results),
                'f1_score': np.mean(f1_results),
            },
            'roc_auc': np.mean(roc_auc_results) if roc_auc_results else np.nan,
            'brier_score': np.mean(brier_score_results),
            'log_loss': np.mean(log_loss_results) if log_loss_results else np.nan
        }
        
        # Compute confusion matrix
        confusion = confusion_matrix(all_y_true, all_y_pred)

        # Calculate the time taken for the whole cross-validation process
        time_taken = time.time() - start_time
        
        # Return metrics, confusion matrix, and time taken
        return {
            'metrics': metrics,
            'confusion_matrix': confusion,
            'time_taken': time_taken
        }
    else:
        print("No valid folds were processed.")
        return None