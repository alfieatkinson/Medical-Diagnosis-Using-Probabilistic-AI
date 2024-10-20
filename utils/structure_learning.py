import pandas as pd
import bnlearn

def learn_structure(train_data : pd.DataFrame, use_bnlearn : bool = False, methodtype : str = 'hc', scoretype : str = 'bic') -> dict[str, list]:
    """Learns the structure of a Bayesian Network given training data.

    Parameters:
        train_data (pd.DataFrame): The training data to learn the structure of.
        use_bnlearn (bool, optional): Whether to use bnlearns structure learning or self-implemented methods. Defaults to False.
        methodtype (str, optional): The method of structure learning to use. Defaults to 'hc'.
        scoretype (str, optional): The scoring method to use. Defaults to 'bic'.
    """
    if use_bnlearn:
        return bnlearn.structure_learning.fit(train_data, methodtype=methodtype, scoretype=scoretype)
    
    return None #TODO - Implement own structure learning algorithms.