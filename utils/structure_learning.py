import pandas as pd
import bnlearn
import itertools
import numpy as np

def bic_score(data : pd.DataFrame, model : dict[str, list]) -> float:
    """Calculates the BIC score for the given data and model.

    Parameters:
        data (pd.DataFrame): The data to score.
        model (dict[str, list]): The model to score.

    Returns:
        float: The calculated score.
    """
    n = len(data)
    likelihood = 0

    for node in model:
        parents = model[node]
        if parents:
            # Calculate conditional probabilities
            cpt = pd.crosstab(index=data[node], columns=[data[parent] for parent in parents], normalize='columns')
        else:
            # If no parents, calculate marginal probabilities
            cpt = pd.crosstab(index=data[node], columns="count", normalize='columns')
        for idx in cpt.index:
            for col in cpt.columns:
                prob = cpt.loc[idx, col]
                if prob > 0:
                    likelihood += prob * np.log(prob) * n

    k = sum(len(parents) for parents in model.values())

    return likelihood - (k / 2) * np.log(n)

def get_score(data : pd.DataFrame, model : dict[str, list], scoretype : str = 'bic') -> float:
    """Scores the given data and model.

    Parameters:
        data (pd.DataFrame): The data to score.
        model (dict[str, list]): The model to score.
        scoretype (str, optional): The scoring method to use. Defaults to 'bic'.

    Returns:
        float: The calculated score.
    """
    if scoretype == 'bic':
        return bic_score(data, model)
    
    if scoretype == 'aic':
        pass #TODO - Implement AIC scoring
    
    raise NotImplementedError(f"Scoring method '{scoretype}' is not implemented yet or does not exist.")

def learn_structure(train_data : pd.DataFrame, use_bnlearn : bool = False, methodtype : str = 'hc', scoretype : str = 'bic') -> dict[str, list]:
    """Learns the structure of a Bayesian Network given training data.

    Parameters:
        train_data (pd.DataFrame): The training data to learn the structure of.
        use_bnlearn (bool, optional): Whether to use bnlearns structure learning or self-implemented methods. Defaults to False.
        methodtype (str, optional): The method of structure learning to use. Defaults to 'hc'.
        scoretype (str, optional): The scoring method to use. Defaults to 'bic'.
        
    Returns:
        model (dict[str, list]): A dictionary containing the model information.
    """
    if use_bnlearn:
        return bnlearn.structure_learning.fit(train_data, methodtype=methodtype, scoretype=scoretype)
    
    if methodtype == 'hc':
        nodes = list(train_data.columns)
        best_model = {node: [] for node in nodes}
        best_score = get_score(train_data, best_model, scoretype=scoretype)
        print(f"Initial Score: {best_score}")

        improved = True
        while improved:
            improved = False
            for node1, node2 in itertools.permutations(nodes, 2):
                if node2 not in best_model[node1]:
                    # Add edge
                    candidate_model = {node: parents.copy() for node, parents in best_model.items()}
                    candidate_model[node1].append(node2)
                    candidate_score = get_score(train_data, candidate_model, scoretype=scoretype)
                    
                    if candidate_score > best_score:
                        best_model = candidate_model
                        best_score = candidate_score
                        improved = True
                        print(f"Improved Score: {best_score} by adding edge {node1} -> {node2}")

        return best_model
    
    raise NotImplementedError(f"Method '{methodtype}' is not implemented yet or does not exist.")