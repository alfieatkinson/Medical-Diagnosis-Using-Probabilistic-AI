import bnlearn
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency

# Step 1: Conditional Independence Test (PC algorithm)
def conditional_independence_test(data, node1, node2, conditioned_set):
    # Perform Chi-squared test for conditional independence
    sub_data = data[[node1, node2] + conditioned_set]
    contingency_table = pd.crosstab(sub_data[node1], sub_data[node2])
    
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value > 0.05  # If p-value > 0.05, consider them independent

# Step 2: PC Algorithm (find the skeleton)
def pc_algorithm(data):
    nodes = list(data.columns)
    skeleton = {node: set() for node in nodes}
    
    # Test for conditional independence for each pair of nodes
    for node1, node2 in combinations(nodes, 2):
        conditional_set = [n for n in nodes if n not in [node1, node2]]
        
        # Conditional independence test
        if conditional_independence_test(data, node1, node2, conditional_set):
            skeleton[node1].add(node2)
            skeleton[node2].add(node1)
    
    return skeleton

# Step 3: Create candidate parent sets
def create_candidate_parents(skeleton, data):
    candidate_parents = {}
    
    for node in skeleton:
        candidate_parents[node] = set(skeleton[node])  # Initially consider all neighbors as candidates
    return candidate_parents

# Step 4: Calculate BIC score for a given structure (model)
def calculate_bic(data, structure):
    # Create a BNlearn model
    model = bnlearn.structure_learning.fit(data, methodtype='hc', max_iter=100, threshold=0.5)
    
    # Compute the BIC score
    return model.score_bic(data)

# Step 5: Hill Climbing Algorithm for Structure Optimisation
def hill_climbing(data, candidate_parents, max_iter=100, scoring_function='bic'):
    # Initialize an empty model
    model = bnlearn.structure_learning.fit(data, methodtype='hc', max_iter=0)  # Start with an empty model (0 iterations)
    
    best_score = float('inf') if scoring_function in ['bic', 'aic'] else -float('inf')
    best_structure = model

    for iteration in range(max_iter):
        improved = False
        for node, candidates in candidate_parents.items():
            # Try adding/removing edges to maximize the score
            for candidate in candidates:
                # Test adding an edge (candidate -> node)
                new_structure = model.copy()
                new_structure.add_edge((candidate, node))
                
                # Calculate the score
                score = calculate_bic(data, new_structure)
                
                # If score improves, update the model
                if score < best_score:  # For BIC and AIC, lower is better
                    best_score = score
                    best_structure = new_structure
                    improved = True
        
        # If no improvement, stop
        if not improved:
            break

    return best_structure

# Step 6: Full MMHC Algorithm (PC + HC)
def mmhc(data, max_iter=100, scoring_function='bic'):
    # Step 1: Run the PC Algorithm to get the skeleton
    skeleton = pc_algorithm(data)
    
    # Step 2: Create candidate parents from the skeleton
    candidate_parents = create_candidate_parents(skeleton, data)
    
    # Step 3: Apply Hill Climbing to optimise the structure
    final_model = hill_climbing(data, candidate_parents, max_iter=max_iter, scoring_function=scoring_function)
    
    return final_model

