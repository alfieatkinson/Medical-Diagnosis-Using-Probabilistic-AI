import pandas as pd 
import bnlearn

def learn_parameters(model : dict[str, list], train_data : pd.DataFrame, use_bnlearn : bool = False, methodtype : str = 'maximumlikelihood', smoothing : str = 'laplace') -> dict[str, list]:
    """Learns the parameters of the given model.

    Parameters:
        model (dict[str, list]): The model to edit.
        train_data (pd.DataFrame): The training data to learn the parameters of.
        use_bnlearn (bool, optional): Whether to use bnlearns structure learning or self-implemented methods. Defaults to False.
        methodtype (str, optional): The method of parameter learning to use. Defaults to 'maximumlikelihood'.
        smoothing (str, optional): The smoothing method to use. Defaults to 'laplace'.

    Returns:
        model (dict[str, list]): A dictionary containing the model information.
    """
    if use_bnlearn:
        return bnlearn.parameter_learning.fit(model, train_data, methodtype=methodtype, smoothing=smoothing)
    
    return #TODO - Implement own parameter learning algorithms.