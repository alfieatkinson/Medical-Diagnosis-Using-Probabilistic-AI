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
    
def plot_confusion_matrix(confusion: list[int], cmap='Blues'):
    """Plot confusion matrix using seaborn heatmap.

    Parameters:
        confusion (list[int]): The confusion matrix to be plotted.
        cmap: The cmap to use.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
def plot_metrics(metrics: dict[str, float], title: str):
    """
    Plot performance metrics (accuracy, precision, recall, F1 score) as bar charts.

    Parameters:
        metrics (dict[str, float]): Dictionary containing performance metrics (e.g., accuracy, precision, recall, f1_score).
        title (str): Title of the plot.
    """
    # Extracting metric values
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Creating the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color=['#800080', '#8A2BE2', '#FF69B4', '#DA70D6'])
    
    # Adding title and labels
    plt.title(f"Model Metrics: {title}")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    
    # Displaying the plot
    plt.ylim(0, 1)  # Since these are probabilities (between 0 and 1)
    plt.show()
    
def plot_confusion_matrices(confusion_dict: dict[str, list[int]], max_plots_per_row=4, cmap='Blues'):
    """
    Plots confusion matrices for each model in the confusion_dict.

    Parameters:
        confusion_dict (dict[str, list[int]]): key: title (str), value: confusion matrix (2D list or numpy array).
        max_plots_per_row (int): maximum number of confusion matrices per row in the plot grid.
        cmap (str): colormap to use for the heatmap.
    """
    num_plots = len(confusion_dict)
    num_rows = (num_plots + max_plots_per_row - 1) // max_plots_per_row  # Calculate the number of rows needed

    # Create a figure with subplots
    plt.figure(figsize=(max_plots_per_row * 5, num_rows * 5))

    # Iterate over the dictionary and plot each confusion matrix
    for i, (title, confusion) in enumerate(confusion_dict.items()):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        ax = plt.subplot(num_rows, max_plots_per_row, i + 1)

        # Plot the confusion matrix using a heatmap
        sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()
    
def plot_metrics_graphs(metrics_dict: dict[str, dict[str, float]], max_plots_per_row=4):
    """
    Plots performance metrics (accuracy, precision, recall, F1 score) as bar charts for each model.

    Parameters:
        metrics_dict (dict[str, dict[str, float]]): key: title (str), value: metrics dictionary (accuracy, precision, recall, f1_score)
        max_plots_per_row (int): maximum number of metric plots per row in the plot grid
    """
    num_plots = len(metrics_dict)
    num_rows = (num_plots + max_plots_per_row - 1) // max_plots_per_row  # Calculate the number of rows needed

    # Create a figure with subplots
    plt.figure(figsize=(max_plots_per_row * 5, num_rows * 5))

    # Iterate over the dictionary and plot each metric graph
    for i, (title, metrics) in enumerate(metrics_dict.items()):
        row = i // max_plots_per_row
        col = i % max_plots_per_row
        ax = plt.subplot(num_rows, max_plots_per_row, i + 1)

        # Extracting metric names and values
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Creating the bar chart
        ax.bar(metric_names, metric_values, color=['#800080', '#8A2BE2', '#FF69B4', '#DA70D6'])
        
        # Adding title and labels
        ax.set_title(title)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)  # Since these are probabilities (between 0 and 1)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()