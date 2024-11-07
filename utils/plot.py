import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
def plot_bn_models(models: dict, max_columns: int = 4):
    """Plot multiple Bayesian Network models side by side using networkx and matplotlib.

    Parameters:
        models (dict): Dictionary where keys are titles and values are bnlearn models to plot.
        max_columns (int): Maximum number of columns for subplots in a row.
    """
    num_models = len(models)
    
    if num_models < 1:
        return 0
    
    # Calculate the number of rows needed for subplots
    num_rows = (num_models + max_columns - 1) // max_columns
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, max_columns, figsize=(5 * max_columns, 5 * num_rows))
    
    # If there's only one plot, axes will be a single object, not an array.
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten axes to make iteration easier if it's an array

    # Iterate over the models and plot them in the subplots
    for i, (title, model) in enumerate(models.items()):
        ax = axes[i]  # Get the current axis for the subplot
        
        # Extract the DAG object from the model
        try:
            dag = model['model']
        except KeyError:
            dag = model
        
        # Get the edges from the DAG and convert to a NetworkX graph
        G = nx.DiGraph(dag.edges())  # Extract edges from the DAG object
        
        # Draw the graph on the axis
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, ax=ax, node_size=500, node_color='#FF69B4', font_size=10, font_weight='bold', arrows=True)
        
        ax.set_title(title)

    # Remove any unused subplots (if any)
    for j in range(num_models, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent overlapping and show the plot
    plt.tight_layout(pad=2.0)
    plt.show()
    
def plot_confusion_matrices(predictions_dict, y_true, max_plots_per_row=4, cmap='Blues'):
    """
    Plots confusion matrices for each prediction in the predictions dictionary.

    Args:
    predictions_dict: dict, key: title (str), value: list of predictions (list of integers where 0 is Non-Demented, 1 is Demented)
    y_true: list or pandas Series, true labels (0 or 1 for Non-Demented or Demented)
    max_plots_per_row: int, maximum number of confusion matrices per row in the plot grid
    """
    num_plots = len(predictions_dict)
    num_rows = (num_plots + max_plots_per_row - 1) // max_plots_per_row  # Calculate the number of rows needed

    # Create a figure with subplots
    plt.figure(figsize=(max_plots_per_row * 5, num_rows * 5))

    # Iterate over the dictionary and plot each confusion matrix
    for i, (title, predictions) in enumerate(predictions_dict.items()):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        ax = plt.subplot(num_rows, max_plots_per_row, i + 1)

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, predictions)

        # Plot the confusion matrix using a heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=['Non-Demented', 'Demented'], yticklabels=['Non-Demented', 'Demented'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()