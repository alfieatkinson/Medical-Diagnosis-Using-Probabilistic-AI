import pandas as pd 
import numpy as np 

def sturges_formula(n: int) -> int:
    """Calculate the number of bins for a histogram using Sturges' formula.

    Parameters:
        n (int): The number of data points.

    Returns:
        int: The calculated number of bins.
    """
    return int(np.ceil(np.log2(n) + 1))

def freedman_diaconis_rule(data: np.ndarray) -> int:
    """Calculate the number of bins for a histogram using the Freedman-Diaconis rule.

    Parameters:
        data (np.ndarray): The input data from which to calculate the bins.

    Returns:
        int: The calculated number of bins.
    """
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    nbins = int(np.ceil((data.max() - data.min()) / bin_width))
    return nbins

def discretise(df: pd.DataFrame, method: str = 'sturges', nbins: int = None) -> pd.DataFrame:
    """Discretise numerical columns in a DataFrame into bins based on the specified method.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical columns to discretise.
        method (str): The method to use for determining the number of bins ('sturges' or 'freedman-diaconis'). Defaults to 'sturges'.
        nbins (int, optional): The number of bins to use. If specified, it overrides the method selection. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with discretised numerical columns.
    
    Raises:
        ValueError: If an invalid method is specified.
    """
    if nbins is not None:
        for col in df.select_dtypes(include=[np.number]).columns:
            # Use the specified number of bins for discretisation
            df[col] = pd.cut(df[col], bins=nbins, labels=[f'Bin{i+1}' for i in range(nbins)])
        return df
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if method == 'sturges':
            bins = sturges_formula(len(df[col]))
        elif method == 'freedman-diaconis':
            bins = freedman_diaconis_rule(df[col])
        else:
            raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
        df[col] = pd.cut(df[col], bins=bins, labels=[f'Bin{i+1}' for i in range(bins)])
    
    return df