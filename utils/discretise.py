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
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # If the data is constant (zero variance), return a default number of bins
    if len(np.unique(data)) == 1:
        print("Warning: Column has zero variance. Returning a default number of bins.")
        return 10  # default value, you can adjust it
    
    # Calculate the IQR
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25

    # Prevent division by zero if the IQR is zero
    if iqr == 0:
        print("Warning: IQR is zero. Returning a default number of bins.")
        return 10  # default value, you can adjust it

    # Calculate the bin width
    bin_width = 2 * iqr * len(data) ** (-1/3)

    # Calculate the number of bins
    nbins = int(np.ceil((data.max() - data.min()) / bin_width))
    
    # Ensure at least one bin
    nbins = max(nbins, 1)

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
        unique_vals = column.nunique()
        if num_bins > unique_vals:
            num_bins = unique_vals
        return pd.cut(column, bins=num_bins, labels=False)

    df_copy = df.copy()

    # Skip target variable (Status) and only discretise features
    feature_columns = [col for col in df_copy.columns if col != 'Status' and col != 'Group']
    
    if nbins is not None:
        for col in feature_columns:
            if df_copy[col].nunique() > 1:  # Skip columns with only one unique value
                df_copy[col] = assign_bins(df_copy[col], nbins)
        return df_copy
    
    # Apply the chosen binning method
    for col in feature_columns:
        if df_copy[col].nunique() <= 1:  # Skip columns with only one unique value
            continue
        
        if df_copy[col].nunique() <= 2:  # Skip columns with very few unique values (e.g., binary columns)
            continue
        
        if method == 'sturges':
            num_bins = sturges_formula(len(df_copy[col]))
        elif method == 'freedman-diaconis':
            num_bins = freedman_diaconis_rule(df_copy[col])
        else:
            raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
        
        df_copy[col] = assign_bins(df_copy[col], num_bins)
    
    return df_copy

def discretise_query(query, df, method):
    for col in query:
        if col in df.columns:
            # Skip columns that are categorical (like 'Status') or already discrete
            if df[col].dtype == 'object' or len(df[col].unique()) <= 2:  # Categorical or binary
                continue
            
            if method == 'sturges':
                num_bins = sturges_formula(len(df[col]))
            elif method == 'freedman-diaconis':
                num_bins = freedman_diaconis_rule(df[col])
            else:
                raise ValueError("Method must be 'sturges' or 'freedman-diaconis'.")
            
            # Discretise the query value based on the number of bins for that column
            query[col] = pd.cut([query[col]], bins=num_bins, labels=False)[0]
    
    return query