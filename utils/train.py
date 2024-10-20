import pandas as pd
import bnlearn
from sklearn.model_selection import train_test_split

def get_train_test_data(df : pd.DataFrame, target : str, test_size : float | int = 0.2, random_state : int = 42) -> tuple[pd.DataFrame, tuple]:
    """Function to generate train and test data for inference with Bayesian Networks.

    Parameters:
        df (DataFrame): A pandas dataframe containing the data.
        target (str): The name of the target column.
        test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. Defaults to 0.2.
        random_state (int): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. Defaults to 42.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = (X_test, y_test)
    
    return train_data, test_data