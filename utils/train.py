import warnings
import logging
import pandas as pd 
import numpy as np 
import bnlearn as bn
from pgmpy.global_vars import logger
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
logger.setLevel(logging.ERROR)
logging.getLogger('seaborn').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

def discrete_cross_validation(df: pd.DataFrame, target: str, nsplits: int = 5, structure_kwargs: dict = None, parameter_kwargs: dict = None) -> dict[str, any]:
    """
    Perform k-fold cross-validation on a Bayesian Network using structure and parameter learning functions.
    
    Parameters:
        df (DataFrame): The dataframe containing the data for the Bayesian Network.
        target (str): The target column name.
        nsplits (int): The number of splits for cross-validation. Defaults to 5.
        structure_kwargs (dict): Additional arguments for the structure learning function.
        parameter_kwargs (dict): Additional arguments for the parameter learning function.
    
    Returns:
        A dictionary of average performance metrics (e.g., accuracy, precision, recall, F1 score).
        True labels and predicted labels for all folds, which can be used for confusion matrix.
    """
    # Setup k-fold cross-validation
    kf = KFold(n_splits=nsplits)
    
    # Initialise lists to hold metric results
    accuracy_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    
    # Lists to store true and predicted values for all folds
    all_y_true = []
    all_y_pred = []
    
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{nsplits}")
        
        # Split data into training and test sets
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        
        # Structure learning
        if structure_kwargs is None:
            structure_kwargs = {}
        model = bn.structure_learning.fit(train_data, **structure_kwargs)
        
        # Parameter learning
        if parameter_kwargs is None:
            parameter_kwargs = {}
        model = bn.parameter_learning.fit(model, train_data, **parameter_kwargs)
        
        # Drop columns from test data not in the model nodes
        model_nodes = model['model'].nodes()
        test_columns = test_data.columns.tolist()
        columns_to_drop = [col for col in test_columns if col not in model_nodes]
        test_data_filtered = test_data.drop(columns=columns_to_drop)
        
        # Evaluate model on the test set
        y_true = test_data[target]
        y_pred = bn.predict(model, test_data_filtered, target) #FIXME - KeyError from bnlearn
        y_pred = np.argmax(y_pred, axis=1)
        
        print("Predicted Values:", y_pred)
        
        # Collect true and predicted values for confusion matrix
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Compute evaluation metrics
        accuracy_results.append(accuracy_score(y_true, y_pred))
        precision_results.append(precision_score(y_true, y_pred, average='binary'))
        recall_results.append(recall_score(y_true, y_pred, average='binary'))
        f1_results.append(f1_score(y_true, y_pred, average='binary'))
    
    # Return average metrics across folds
    metrics = {
        'accuracy': sum(accuracy_results) / nsplits,
        'precision': sum(precision_results) / nsplits,
        'recall': sum(recall_results) / nsplits,
        'f1_score': sum(f1_results) / nsplits
    }
    
    # Return metrics and confusion matrix data (true and predicted values)
    confusion = confusion_matrix(all_y_true, all_y_pred)
    
    return metrics, confusion