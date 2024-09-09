import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


def plot_training_history(history: Any, width: int = 1000, height: int = 400) -> None:
    """
    Plot the training and validation loss and F1 score from the training history.

    Args:
    history (Any): The training history object returned by Keras.
    width (int, optional): The width of the plot. Defaults to 1000.
    height (int, optional): The height of the plot. Defaults to 400.
    """
    # Extract the history dictionary
    history_dict = history.history

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss Evolution', 'F1_score Evolution'))

    # Add traces for training and validation loss (left subplot)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history_dict['loss']) + 1)),
        y=history_dict['loss'],
        mode='lines+markers',
        name='Train Loss'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(1, len(history_dict['val_loss']) + 1)),
        y=history_dict['val_loss'],
        mode='lines+markers',
        name='Val Loss'
    ), row=1, col=1)

    # Add traces for training and validation F1 score (right subplot)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history_dict['f1_score']) + 1)),
        y=history_dict['f1_score'],
        mode='lines+markers',
        name='Train f1_score'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=list(range(1, len(history_dict['val_f1_score']) + 1)),
        y=history_dict['val_f1_score'],
        mode='lines+markers',
        name='Val f1_score'
    ), row=1, col=2)

    # Update layout for the entire figure
    fig.update_layout(
        width=width,  # Set the width of the figure
        height=height,
        showlegend=True,
        legend=dict(x=1.05, y=1, orientation='v'),  # Place legend on the right
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Remove background
        paper_bgcolor='rgba(0,0,0,0)',  # Remove background
        font=dict(color='black')  # Set font color to black
    )

    # Update x-axis and y-axis titles for each subplot
    fig.update_xaxes(title_text='Epochs', row=1, col=1, title_font=dict(color='black'), tickfont=dict(color='black'))
    fig.update_yaxes(title_text='Loss', row=1, col=1, title_font=dict(color='black'), tickfont=dict(color='black'), range=[0, 1])
    fig.update_xaxes(title_text='Epochs', row=1, col=2, title_font=dict(color='black'), tickfont=dict(color='black'))
    fig.update_yaxes(title_text='f1_score', row=1, col=2, title_font=dict(color='black'), tickfont=dict(color='black'), range=[0, 1])

    # Show the plot
    fig.show()
    

def plot_f1_scores_vs_thresholds(thresholds: List[float], f1_scores: List[float], best_threshold: float, width: int = 1000, height: int = 400) -> None:
    """
    Plot F1 scores vs thresholds and highlight the best threshold.

    Args:
    thresholds (List[float]): List of threshold values.
    f1_scores (List[float]): List of F1 scores corresponding to the thresholds.
    best_threshold (float): The best threshold value.
    width (int, optional): The width of the plot. Defaults to 1000.
    height (int, optional): The height of the plot. Defaults to 400.
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add a line plot for F1 scores vs thresholds
    fig.add_trace(go.Scatter(
        x=thresholds, 
        y=f1_scores, 
        mode='lines+markers', 
        name='F1 Score'
    ))

    # Add a vertical line for the best threshold
    fig.add_trace(go.Scatter(
        x=[best_threshold, best_threshold], 
        y=[0, max(f1_scores)], 
        mode='lines', 
        name='Best Threshold', 
        line=dict(dash='dash')
    ))

    # Update layout for the entire figure
    fig.update_layout(
        title='F1 Score vs Threshold',
        width=width,  # Set the width of the figure
        height=height,
        showlegend=True,
        legend=dict(x=1.05, y=1, orientation='v'),  # Place legend on the right
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Remove background
        paper_bgcolor='rgba(0,0,0,0)',  # Remove background
        font=dict(color='black')  # Set font color to black
    )

    # Update x-axis and y-axis titles
    fig.update_xaxes(title_text='Threshold', title_font=dict(color='black'), tickfont=dict(color='black'))
    fig.update_yaxes(title_text='F1 Score', title_font=dict(color='black'), tickfont=dict(color='black'))

    # Show the plot
    fig.show()



def plot_confusion_matrix(conf_matrix: Any, figsize: tuple = (8, 6), cmap: str = 'viridis') -> None:
    """
    Plot the confusion matrix using Seaborn.

    Args:
    conf_matrix (Any): The confusion matrix to be plotted.
    figsize (tuple, optional): The size of the figure. Defaults to (8, 6).
    cmap (str, optional): The colormap to be used. Defaults to 'viridis'.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()



def plot_roc_curve(y_true: List[int], y_pred_prob: List[float], width: int = 1000, height: int = 400) -> None:
    """
    Plot the ROC curve using Plotly.

    Args:
    y_true (List[int]): True binary labels.
    y_pred_prob (List[float]): Target scores, can either be probability estimates of the positive class or confidence values.
    width (int, optional): The width of the plot. Defaults to 1000.
    height (int, optional): The height of the plot. Defaults to 400.
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve plot
    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (area = {roc_auc:.2f})',
        line=dict(color='red', width=2)
    ))

    # Add a diagonal line representing a random classifier
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # Update layout for the ROC curve plot
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=width,  # Set the width of the figure
        height=height,  # Set the height of the figure
        showlegend=True,
        legend=dict(x=1.05, y=1, orientation='v'),  # Place legend on the right
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Remove background
        paper_bgcolor='rgba(0,0,0,0)',  # Remove background
        font=dict(color='black')  # Set font color to black
    )

    # Show the ROC curve plot
    fig_roc.show()