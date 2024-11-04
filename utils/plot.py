import networkx as nx
import matplotlib.pyplot as plt

def plot_directed_graph(model : dict[str, list]):
    """Plots a directed graph given a model.

    Parameters:
        model (dict[str, list]): The model to plot.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Assuming your model has a structure like this:
    # model = {'node1': ['node2', 'node3'], 'node2': ['node4'], ...}
    for parent, children in model.items():
        for child in children:
            G.add_edge(parent, child)

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, arrows=True)
    
    # Display the graph
    plt.show()