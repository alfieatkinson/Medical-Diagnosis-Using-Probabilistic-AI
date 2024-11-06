import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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
    
def plot_bins_barcharts(df: pd.DataFrame):
    """Plot bar charts for all columns in the DataFrame with custom x-ticks based on bin labels.

    Parameters:
        df (pd.DataFrame): The input DataFrame with binned categorical columns.
    """
    # Determine the layout of the subplots
    num_columns = len(df.columns)
    num_rows = (num_columns + 3) // 4  # Ensures enough rows for up to 4 columns per row

    # Create subplots
    fig, axes = plt.subplots(num_rows, 4, figsize=(25, 20))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, col in zip(axes, df.columns):
        # Get the value counts for each column
        value_counts = df[col].value_counts().sort_index()
        
        # Plot bar chart
        ax.bar(value_counts.index.astype(str), value_counts.values)
        ax.set_title(col)
        ax.set_xlabel('Bins')
        ax.set_ylabel('Frequency')
        
        # Customize x-ticks
        num_bins = len(value_counts)
        ax.set_xticks(range(num_bins))
        ax.set_xticklabels([f'Bin {i+1}' for i in range(num_bins)])

    # Remove any unused subplots
    for ax in axes[num_columns:]:
        ax.axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()