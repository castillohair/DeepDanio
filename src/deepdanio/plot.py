import matplotlib
import numpy
import pandas
from matplotlib import pyplot
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import CubicSpline

from deepdanio import definitions

# Default color of each lineage
LINEAGE_COLORS = {
    'neural ectoderm': 'darkgreen',
    'non-neural ectoderm': 'darkred',
    'mesoderm': 'darkgoldenrod',
    'endoderm': 'tab:blue',
    'ysl': 'tab:pink',
    'evl': 'tab:orange',
    'other': 'k',
}


def bar(
        cell_state_vals,
        cell_states=None,
        cell_state_colors=None,
        width=0.8,
        stage_title_fontsize='small',
        stage_title_y_offset=0.02,
        ylim=None,
        figsize=None,
        ax=None,
    ):
    """
    Bar plot of values across cell states, grouped by stage.

    Stages are separated by dashed lines and labeled above the plot.

    Parameters
    ----------
    cell_state_vals : array-like
        Values with shape (n_cell_states,), or (n_samples, n_cell_states) to
        plot the mean with the standard deviation as error bars.
    cell_states : list of str, optional
        Cell states corresponding to the values, ordered by stage. Default is
        all cell states in model output order.
    cell_state_colors : dict, optional
        Bar color of each cell state. Default is the color of its lineage.
    width : float, optional
        Bar width.
    stage_title_fontsize : str or float, optional
        Font size of stage labels.
    stage_title_y_offset : float, optional
        Vertical offset of stage labels above the plot, as a fraction of the
        y axis range.
    ylim : tuple, optional
        Y axis limits.
    figsize : tuple, optional
        Figure size, if a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    """
    if cell_states is None:
        cell_states = definitions.CELL_STATES
    metadata = definitions.CELL_STATE_METADATA.loc[cell_states]
    if cell_state_colors is None:
        cell_state_colors = metadata['lineage'].map(LINEAGE_COLORS).to_dict()

    # Bar positions, with a gap between stages
    gap_size = 1
    stages = metadata['stage'].values
    is_new_stage = numpy.concatenate([[False], stages[1:] != stages[:-1]])
    x = numpy.arange(len(cell_states)) + gap_size * numpy.cumsum(is_new_stage)

    # Stage dividers and label positions
    stage_start_idx = numpy.flatnonzero(numpy.concatenate([[True], is_new_stage[1:]]))
    x_divs = numpy.concatenate([x[stage_start_idx] - (gap_size + 1) / 2, [x[-1] + (gap_size + 1) / 2]])
    stage_labels_x = (x_divs[:-1] + x_divs[1:]) / 2
    stage_labels = stages[stage_start_idx]

    if ax is None:
        if figsize is None:
            figsize = (len(cell_states) / 64 * 10, 4)
        fig, ax = pyplot.subplots(figsize=figsize)

    cell_state_vals = numpy.squeeze(numpy.asarray(cell_state_vals, dtype=float))
    colors = [cell_state_colors.get(c, LINEAGE_COLORS['other']) for c in cell_states]
    if cell_state_vals.ndim == 1:
        ax.bar(x, cell_state_vals, color=colors, width=width)
    else:
        ax.bar(x, cell_state_vals.mean(axis=0), yerr=cell_state_vals.std(axis=0), color=colors, width=width)

    ax.set_xticks(x)
    ax.set_xticklabels(metadata['cell_type'], rotation=90, fontsize='small')
    ax.set_xlim(x_divs[0], x_divs[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    for x_div in x_divs:
        ax.axvline(x_div, color='k', linestyle='--', linewidth=1)

    y_label = ax.get_ylim()[1] + stage_title_y_offset * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for stage_label, stage_label_x in zip(stage_labels, stage_labels_x):
        ax.text(
            stage_label_x,
            y_label,
            stage_label,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=stage_title_fontsize,
        )

    return ax


def trajectory(
        cell_state_vals=None,
        cmap='viridis',
        vmin=None,
        vmax=None,
        colorbar=False,
        colorbar_label=None,
        cell_state_colors=None,
        shade_edges=True,
        cell_state_labels=None,
        stage_labels=True,
        markersize=25,
        linewidth=1,
        cell_state_label_size=8,
        stage_label_size=8,
        figsize=(8, 6),
        ax=None,
    ):
    """
    Plot the cell state differentiation trajectory.

    Cell states are shown as dots, positioned by stage (x axis) and lineage
    (y axis), connected by edges with opacity given by their probability.
    Dots are colored by value using a colormap if values are provided, or by
    lineage otherwise.

    Parameters
    ----------
    cell_state_vals : dict, pandas.Series, or array-like, optional
        Value of each cell state. Arrays must follow the model output order.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used to color cell states by value.
    vmin, vmax : float, optional
        Values mapped to the ends of the colormap. Default is the minimum and
        maximum of the values.
    colorbar : bool, optional
        Whether to draw a vertical colorbar to the right of the plot and cell
        state labels. Requires cell_state_vals. The colorbar is outside the
        axes, so leave space for it next to other subplots, and use
        bbox_inches='tight' when saving.
    colorbar_label : str, optional
        Colorbar label.
    cell_state_colors : dict, optional
        Color of each cell state, used if no values are provided. Default is
        the color of its lineage.
    shade_edges : bool, optional
        Whether to color edges with a gradient between the colors of the cell
        states they connect. If False, edges are black.
    cell_state_labels : {None, 'all', 'final'}, optional
        Label all cell states, only those in the last stage, or none.
    stage_labels : bool, optional
        Whether to label each stage below its cell states. Labels are x axis
        tick labels, so setting x ticks afterwards replaces them.
    markersize : float, optional
        Size of cell state markers.
    linewidth : float, optional
        Width of edges.
    cell_state_label_size : float, optional
        Font size of cell state labels.
    stage_label_size : float, optional
        Font size of stage labels.
    figsize : tuple, optional
        Figure size, if a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    """
    metadata = definitions.CELL_STATE_METADATA
    cell_states = definitions.CELL_STATES
    coords_df = pandas.read_csv(definitions.TRAJECTORY_COORDS_PATH, sep='\t', index_col='cell_state')
    edges_df = pandas.read_csv(definitions.TRAJECTORY_EDGES_PATH, sep='\t')
    if colorbar and cell_state_vals is None:
        raise ValueError("A colorbar requires cell_state_vals.")

    # Cell state colors
    if cell_state_vals is not None:
        if isinstance(cell_state_vals, (dict, pandas.Series)):
            cell_state_vals = [cell_state_vals[c] for c in cell_states]
        cell_state_vals = numpy.asarray(cell_state_vals, dtype=float)
        if vmin is None:
            vmin = cell_state_vals.min()
        if vmax is None:
            vmax = cell_state_vals.max()
        if vmax > vmin:
            normalized_vals = (cell_state_vals - vmin) / (vmax - vmin)
        else:
            normalized_vals = numpy.full(len(cell_state_vals), 0.5)
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        cell_state_colors = dict(zip(cell_states, cmap(normalized_vals)))
    elif cell_state_colors is None:
        cell_state_colors = metadata['lineage'].map(LINEAGE_COLORS).to_dict()

    if ax is None:
        fig, ax = pyplot.subplots(figsize=figsize)

    # Cell states
    final_stage = definitions.STAGES[-1]
    for cell_state, row in metadata.iterrows():
        x, y = coords_df.loc[cell_state, ['x', 'y']]
        ax.scatter(x, y, color=cell_state_colors[cell_state], s=markersize, zorder=2)
        if cell_state_labels == 'all':
            ax.text(x, y, row['cell_type'], ha='left', va='bottom', fontsize=cell_state_label_size, zorder=3, rotation=45)
        elif cell_state_labels == 'final' and row['stage'] == final_stage:
            ax.text(x + 0.2, y, row['cell_type'], ha='left', va='center', fontsize=cell_state_label_size, zorder=3)

    # Edges, as splines with zero slope at both ends
    for cell_state_1, cell_state_2, edge_prob in zip(
            edges_df['cell_state_1'], edges_df['cell_state_2'], edges_df['edge_prob']):
        x1, y1 = coords_df.loc[cell_state_1, ['x', 'y']]
        x2, y2 = coords_df.loc[cell_state_2, ['x', 'y']]
        x_line = numpy.linspace(x1, x2, 100)
        y_line = CubicSpline([x1, x2], [y1, y2], bc_type='clamped')(x_line)

        if shade_edges:
            # Color line segments individually, from one cell state's color to the other's
            edge_cmap = LinearSegmentedColormap.from_list(
                'edge', [cell_state_colors[cell_state_1], cell_state_colors[cell_state_2]], N=100,
            )
            points = numpy.array([x_line, y_line]).T.reshape(-1, 1, 2)
            segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
            ax.add_collection(LineCollection(
                segments,
                cmap=edge_cmap,
                norm=pyplot.Normalize(0, 1),
                array=numpy.linspace(0, 1, len(segments)),
                linewidth=linewidth,
                alpha=edge_prob,
            ))
        else:
            ax.plot(x_line, y_line, color='black', alpha=edge_prob, lw=1, zorder=1)

    ax.set_xlim(coords_df['x'].min() - 0.5, coords_df['x'].max() + 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    # Stage labels, below the cell states of each stage
    if stage_labels:
        stage_x = []
        for stage in definitions.STAGES:
            stage_x_unique = coords_df.loc[metadata.index[metadata['stage'] == stage], 'x'].unique()
            if len(stage_x_unique) != 1:
                raise ValueError(f"Cell states of stage {stage} have different x coordinates: {stage_x_unique}.")
            stage_x.append(stage_x_unique[0])
        ax.set_xticks(stage_x)
        ax.set_xticklabels(
            definitions.STAGES,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
            fontsize=stage_label_size,
        )
        ax.tick_params(axis='x', length=0)
    else:
        ax.set_xticks([])

    # Colorbar, to the right of the axes and cell state labels
    if colorbar:
        ax.figure.canvas.draw()
        cax_x = 1
        if ax.texts:
            labels_right = max(t.get_window_extent().x1 for t in ax.texts)
            cax_x = max(cax_x, ax.transAxes.inverted().transform((labels_right, 0))[0])
        cax = ax.inset_axes([cax_x + 0.05, 0.25, 0.03, 0.5])
        ax.figure.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=cax, label=colorbar_label)

    return ax
