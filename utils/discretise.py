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
    # If the data is constant (zero variance), return a default number of bins
    if len(np.unique(data)) == 1:
        print("Warning: Column has zero variance. Returning a default number of bins.")
        return 1
    
    # Calculate the IQR
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25

    # Prevent division by zero if the IQR is zero
    if iqr == 0:
        print("Warning: IQR is zero. Returning a default number of bins.")
        return 10

    # Calculate the bin width
    bin_width = 2 * iqr * len(data) ** (-1/3)

    # Calculate the number of bins
    nbins = int(np.ceil((data.max() - data.min()) / bin_width))
    
    # Ensure at least one bin
    nbins = max(nbins, 1)

    return nbins

def assign_bins(column: pd.Series, num_bins: int) -> pd.Series:
    """Assign bins to a numerical column.

    Parameters:
        column (pd.Series): The numerical column to be binned.
        num_bins (int): The number of bins to use.

    Returns:
        pd.Series: The binned column.
    """
    unique_vals = column.nunique()
    if num_bins > unique_vals:
        num_bins = unique_vals
    bin_edges = np.linspace(column.min(), column.max(), num=num_bins+1)
    return pd.cut(column, bins=bin_edges, labels=range(num_bins), include_lowest=True)

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
    df_copy = df.copy()
    feature_columns = df_copy.columns.tolist()
    
    if nbins is not None:
        for col in feature_columns:
            if df_copy[col].nunique() > 1:
                df_copy[col] = assign_bins(df_copy[col], nbins)
        return df_copy
    
    for col in feature_columns:
        if df_copy[col].nunique() <= 1:
            continue
        
        if method == 'sturges':
            num_bins = sturges_formula(len(df_copy[col]))
        elif method == 'freedman-diaconis':
            num_bins = freedman_diaconis_rule(df_copy[col].astype(float).values)
        else:
            raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
        
        df_copy[col] = assign_bins(df_copy[col], num_bins)
    
    return df_copy

def discretise_query(query: dict, df: pd.DataFrame, method: str) -> dict:
    """Discretise the query values based on the binning of the training DataFrame.

    Parameters:
        query (dict): The query values to be discretised.
        df (pd.DataFrame): The training DataFrame used to determine the bins.
        method (str): The method to use for determining the number of bins ('sturges' or 'freedman-diaconis').

    Returns:
        dict: The discretised query values.
    
    Raises:
        ValueError: If an invalid method is specified.
    """
    query_copy = query.copy()
    for col in query:
        if col in df.columns:
            if method == 'sturges':
                num_bins = sturges_formula(len(df[col]))
            elif method == 'freedman-diaconis':
                num_bins = freedman_diaconis_rule(df[col].astype(float).values)
            else:
                raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
            
            bins = np.linspace(df[col].min(), df[col].max(), num=num_bins+1)
            query_copy[col] = pd.cut([query[col]], bins=bins, labels=range(num_bins), include_lowest=True)[0]
    
    return query_copy