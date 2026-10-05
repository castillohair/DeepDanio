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
