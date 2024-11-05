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
        nbins (Optional[int]): The number of bins to use. If specified, it overrides the method selection. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with discretised numerical columns.
    
    Raises:
        ValueError: If an invalid method is specified.
    """
    
    def assign_bins(column: pd.Series, num_bins: int) -> pd.Series:
        """Helper function to assign bins to a numerical column."""
        return pd.cut(column, bins=num_bins, labels=[f'Bin{i+1}' for i in range(num_bins)])

    if nbins is not None:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = assign_bins(df[col], nbins)
        return df
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() == 1:
            df[col] = assign_bins(df[col], 1)
            continue
        
        if method == 'sturges':
            num_bins = sturges_formula(len(df[col]))
        elif method == 'freedman-diaconis':
            num_bins = freedman_diaconis_rule(df[col])
        else:
            raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
        
        df[col] = assign_bins(df[col], num_bins)
    
    return df