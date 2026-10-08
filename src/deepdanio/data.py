from pathlib import Path

import h5py
import numpy
import pandas

from deepdanio import definitions


def quantile_normalize(values):
    """
    Quantile normalize the columns of a matrix.

    Each value is replaced by the mean across columns of the sorted values at
    its rank. Tied values receive the rank of their first occurrence.

    Parameters
    ----------
    values : numpy.ndarray
        Matrix with shape (n_rows, n_columns).

    Returns
    -------
    numpy.ndarray
        Quantile-normalized matrix with the same shape.

    """
    sorted_values = numpy.sort(values, axis=0)
    rank_means = sorted_values.mean(axis=1)
    normalized = numpy.empty_like(values, dtype=float)
    for col_idx in range(values.shape[1]):
        ranks = numpy.searchsorted(sorted_values[:, col_idx], values[:, col_idx], side='left')
        normalized[:, col_idx] = rank_means[ranks]
    return normalized


def load_training_data(group='peaks', filepath=definitions.TRAINING_DATA_PATH):
    """
    Load processed peaks or negative regions.

    Parameters
    ----------
    group : {'peaks', 'negatives'}, optional
        Which set of regions to load.
    filepath : str or pathlib.Path, optional
        Path to the processed HDF5 file.

    Returns
    -------
    metadata_df : pandas.DataFrame
        Columns 'chr', 'start', 'end', and 'sequence', indexed by region ID.
    signal_df : pandas.DataFrame or None
        Accessibility signal (regions x cell states). None for negatives.

    """
    with h5py.File(filepath, 'r') as f:
        h5_group = f[group]
        ids = h5_group['id'].asstr()[:]
        metadata_df = pandas.DataFrame(
            {
                'chr': h5_group['chr'].asstr()[:],
                'start': h5_group['start'][:],
                'end': h5_group['end'][:],
                'sequence': h5_group['sequence'].asstr()[:],
            },
            index=pandas.Index(ids, name='id'),
        )
        if 'signal' in h5_group:
            signal_df = pandas.DataFrame(
                h5_group['signal'][:],
                index=metadata_df.index,
                columns=f['cell_states'].asstr()[:],
            )
        else:
            signal_df = None

    return metadata_df, signal_df


def load_contributions(peak_ids=None, cell_states=None, filepath=definitions.CONTRIBUTIONS_PATH):
    """
    Load actual contribution scores of cell state-specific peaks.

    Parameters
    ----------
    peak_ids : list of str, optional
        Peaks to load. Default is all peaks, which requires about 30 GB of
        memory.
    cell_states : list of str, optional
        Cell states to load. Default is all cell states in model output order.
    filepath : str or pathlib.Path, optional
        Path to the contributions file.

    Returns
    -------
    numpy.ndarray
        Contributions with shape (n_peaks, n_cell_states, seq_length), in the
        order of `peak_ids` and `cell_states`.

    """
    with h5py.File(filepath, 'r') as f:
        dataset = f['contributions']
        file_cell_states = f['cell_state'].asstr()[:].tolist()

        # Rows to read, in increasing order as required by h5py, and their
        # position in the output
        if peak_ids is None:
            rows = slice(None)
            output_rows = slice(None)
            n_peaks = dataset.shape[0]
        else:
            peak_idx = pandas.Index(f['peak_id'].asstr()[:]).get_indexer(peak_ids)
            if (peak_idx < 0).any():
                raise KeyError(f"Peaks not found: {list(numpy.asarray(peak_ids)[peak_idx < 0])}")
            output_rows = numpy.argsort(peak_idx)
            rows = peak_idx[output_rows]
            n_peaks = len(peak_idx)

        # Read whole rows, or only the columns of the requested cell states
        if cell_states is None:
            contribs = numpy.empty((n_peaks,) + dataset.shape[1:], dtype='float32')
            contribs[output_rows] = dataset[rows]
        else:
            contribs = numpy.empty((n_peaks, len(cell_states), dataset.shape[2]), dtype='float32')
            for col_idx, cell_state in enumerate(cell_states):
                contribs[output_rows, col_idx] = dataset[rows, file_cell_states.index(cell_state)]

    return contribs


def load_hypothetical_contributions(cell_state, peak_ids=None, dirpath=definitions.HYPOTHETICAL_CONTRIBUTIONS_DIR):
    """
    Load hypothetical contribution scores of peaks in one cell state.

    Parameters
    ----------
    cell_state : str
        Cell state.
    peak_ids : list of str, optional
        Peaks to load. Default is all peaks in the file.
    dirpath : str or pathlib.Path, optional
        Directory with one hypothetical contributions file per cell state.

    Returns
    -------
    contribs : numpy.ndarray
        Hypothetical contributions with shape (n_peaks, seq_length, 4).
    peak_ids : list of str
        Peaks corresponding to the contributions.

    Raises
    ------
    FileNotFoundError
        If the file of the cell state is not present.

    """
    cell_state_idx = definitions.CELL_STATES.index(cell_state)
    filepath = Path(dirpath) / definitions.HYPOTHETICAL_CONTRIBUTIONS_NAME.format(cell_state_idx=cell_state_idx)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Hypothetical contributions of {cell_state} not found at {filepath}. These files are not "
            "publicly available due to their size. They can be computed with "
            "analysis/motif_discovery/compute_contributions.py, or requested from the authors."
        )

    with h5py.File(filepath, 'r') as f:
        file_peak_ids = pandas.Index(f['peak_id'].asstr()[:])
        if peak_ids is None:
            return f['contributions'][:], file_peak_ids.tolist()
        # h5py requires increasing indices
        peak_idx = file_peak_ids.get_indexer(peak_ids)
        if (peak_idx < 0).any():
            raise KeyError(f"Peaks not found: {list(numpy.asarray(peak_ids)[peak_idx < 0])}")
        sort_order = numpy.argsort(peak_idx)
        contribs = numpy.empty((len(peak_idx),) + f['contributions'].shape[1:], dtype='float32')
        contribs[sort_order] = f['contributions'][peak_idx[sort_order]]

    return contribs, list(peak_ids)
