from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jointsbm.core.models import SBMResults


def plot_connectivity_matrix(
    B: Union[np.ndarray, SBMResults],
    precision=".3f",
    ax=None,
    title="Connectivity Matrix",
    cmap="viridis",
):
    """
    Plot the connectivity matrix B.

    Parameters:
    B (np.ndarray): The connectivity matrix to plot.
    ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
    title (str): The title of the plot.
    cmap (str): The colormap to use for the plot.

    Returns:
    matplotlib.axes.Axes: The axes with the plotted connectivity matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    if isinstance(B, SBMResults):
        B = B.theta
    sns.heatmap(B, annot=True, fmt=precision, cmap=cmap, cbar=True, ax=ax)
    ax.set_title(title)
    return ax
