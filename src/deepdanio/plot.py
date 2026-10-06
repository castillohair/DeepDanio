import logomaker
import matplotlib
import numpy
import pandas
from matplotlib import pyplot
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline

from deepdanio import definitions, sequence

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

# Nucleotide colors for sequence logos
NT_COLORS = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'T': (228/255, 38/255, 56/255),
}

# Lineage names for legends
LINEAGE_LABELS = {
    'neural ectoderm': 'Neural ectoderm',
    'non-neural ectoderm': 'Non-neural ectoderm',
    'mesoderm': 'Mesoderm',
    'endoderm': 'Endoderm',
    'ysl': 'YSL',
    'evl': 'EVL',
    'other': 'Other',
}


def bar(
        cell_state_vals,
        cell_states=None,
        cell_state_colors=None,
        width=0.8,
        stage_title_fontsize='small',
        stage_title_y_offset=0.02,
        ylim=None,
        legend=True,
        figsize=None,
        ax=None,
    ):
    """
    Bar plot of values across cell states, grouped by stage.

    Stages are separated by dashed lines and labeled above the plot. Labels
    that would overlap the previous label are moved up to a higher row.
    Overlaps are evaluated with the axes size at the time of plotting.

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
        axes height. Labels stay above the plot if y limits change later.
    ylim : tuple, optional
        Y axis limits.
    legend : bool, optional
        Whether to show a legend of lineage colors to the right of the plot.
        Only used with the default lineage colors.
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
    lineage_colors_used = cell_state_colors is None
    if lineage_colors_used:
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

    # Stage labels, with x in data coordinates and y in axes coordinates
    stage_texts = []
    for stage_label, stage_label_x in zip(stage_labels, stage_labels_x):
        stage_texts.append(ax.text(
            stage_label_x,
            1 + stage_title_y_offset,
            stage_label,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=stage_title_fontsize,
            transform=ax.get_xaxis_transform(),
        ))

    # Move each label to the lowest row where it does not overlap the
    # previous label in that row, measuring labels in display coordinates
    ax.figure.canvas.draw()
    extents = [t.get_window_extent() for t in stage_texts]
    row_height = max(e.height for e in extents) / ax.get_window_extent().height
    min_label_spacing = extents[0].height / 2
    row_right_edges = []
    for stage_text, extent in zip(stage_texts, extents):
        row = 0
        while row < len(row_right_edges) and extent.x0 < row_right_edges[row] + min_label_spacing:
            row += 1
        if row == len(row_right_edges):
            row_right_edges.append(extent.x1)
        else:
            row_right_edges[row] = extent.x1
        stage_text.set_y(1 + stage_title_y_offset + row * row_height)

    # Legend with the lineages of the plotted cell states
    if legend and lineage_colors_used:
        lineages = [lineage for lineage in LINEAGE_COLORS if lineage in set(metadata['lineage'])]
        handles = [Patch(color=LINEAGE_COLORS[lineage], label=LINEAGE_LABELS[lineage]) for lineage in lineages]
        ax.legend(handles=handles, title='Lineage', loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

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


def sequence_logo(
        nt_height=None,
        pwm=None,
        seq=None,
        first_position=0,
        font_name='DejaVu Sans Mono',
        title=None,
        figsize=None,
        ax=None,
    ):
    """
    Plot a sequence logo.

    The logo is specified by nucleotide heights, a position probability matrix
    (PWM), or a sequence, in that order of preference. PWM letter heights are
    scaled by information content.

    Parameters
    ----------
    nt_height : numpy.ndarray, optional
        Nucleotide heights with shape (length, 4), e.g. contribution scores.
    pwm : numpy.ndarray, optional
        Position probability matrix with shape (length, 4).
    seq : str, optional
        Sequence.
    first_position : int, optional
        X coordinate of the first position.
    font_name : str, optional
        Font of the logo letters.
    title : str, optional
        Axes title.
    figsize : tuple, optional
        Figure size, if a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    """
    if nt_height is None:
        if pwm is not None:
            entropy = numpy.zeros_like(pwm)
            entropy[pwm > 0] = -pwm[pwm > 0] * numpy.log2(pwm[pwm > 0])
            information_content = 2 - entropy.sum(axis=1, keepdims=True)
            nt_height = pwm * information_content
        elif seq is not None:
            nt_height = sequence.one_hot_encode([seq])[0]
        else:
            raise ValueError("One of nt_height, pwm, or seq must be provided.")

    if ax is None:
        if figsize is None:
            figsize = (len(nt_height) / 20, 0.5)
        fig, ax = pyplot.subplots(figsize=figsize)

    logo = logomaker.Logo(
        pandas.DataFrame(
            nt_height,
            columns=['A', 'C', 'G', 'T'],
            index=numpy.arange(first_position, first_position + len(nt_height)),
        ),
        color_scheme=NT_COLORS,
        font_name=font_name,
        ax=ax,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True, linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    return ax


def contribution_logos(
        seq,
        contribs,
        cell_states,
        start=None,
        end=None,
        ylabel_orientation='horizontal',
        figsize=None,
    ):
    """
    Plot a sequence logo and its contribution scores in several cell states.

    Contributions of each cell state are plotted as a logo of the actual
    contributions (hypothetical contributions of the present bases), with a
    common y axis.

    Parameters
    ----------
    seq : str or numpy.ndarray
        Sequence, or one-hot sequence with shape (seq_length, 4).
    contribs : numpy.ndarray
        Hypothetical contributions with shape (n_cell_states, seq_length, 4).
    cell_states : list of str
        Cell state of each contribution array, used as row labels.
    start, end : int, optional
        Region of the sequence to plot. Default is the whole sequence.
    ylabel_orientation : {'horizontal', 'vertical'}, optional
        Orientation of cell state labels.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot.

    """
    seq_onehot = sequence.one_hot_encode([seq])[0] if isinstance(seq, str) else numpy.asarray(seq)
    if len(contribs) != len(cell_states):
        raise ValueError("contribs and cell_states must have the same length.")
    start = 0 if start is None else start
    end = len(seq_onehot) if end is None else end
    seq_onehot = seq_onehot[start:end]
    actual_contribs = numpy.asarray(contribs)[:, start:end] * seq_onehot

    if figsize is None:
        figsize = (max(len(seq_onehot) / 20, 4), 0.5 + 0.5 * len(cell_states))
    fig, axes = pyplot.subplots(1 + len(cell_states), 1, sharex=True, figsize=figsize)

    sequence_logo(nt_height=seq_onehot, first_position=start, ax=axes[0])
    axes[0].spines['bottom'].set_visible(False)

    for ax, cell_state, cell_state_contribs in zip(axes[1:], cell_states, actual_contribs):
        sequence_logo(nt_height=cell_state_contribs, first_position=start, ax=ax)
        ax.spines['left'].set_visible(True)
        if ylabel_orientation == 'horizontal':
            ax.set_ylabel(cell_state, rotation=0, ha='right', va='center')
        else:
            ax.set_ylabel(cell_state)

    # Common y limits across cell states
    ylim = (min(ax.get_ylim()[0] for ax in axes[1:]), max(ax.get_ylim()[1] for ax in axes[1:]))
    for ax in axes[1:]:
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())

    axes[-1].set_xlim(start - 0.5, end - 0.5)
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    axes[-1].set_xlabel('Position (nt)')

    return fig
