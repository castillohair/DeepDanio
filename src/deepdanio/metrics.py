import numpy
import pandas
import scipy.stats


def get_regression_metrics(signal_df, pred_df, per):
    """
    Regression metrics between measured and predicted signal.

    Only pairs where both values are finite are used. Correlations are NaN
    with fewer than two pairs or when either side is constant, and RMSE is
    NaN with no pairs.

    Parameters
    ----------
    signal_df, pred_df : pandas.DataFrame
        Measured and predicted signal, with the same index (regions) and
        columns (cell states).
    per : {'cell_state', 'region'}
        Calculate metrics for each cell state across regions, or for each
        region across cell states.

    Returns
    -------
    pandas.DataFrame
        Columns 'pearson_r', 'spearman_r', 'rmse', and 'n_valid' (number of
        pairs used), indexed by cell state or region.

    """
    if not (signal_df.index.equals(pred_df.index) and signal_df.columns.equals(pred_df.columns)):
        raise ValueError("signal_df and pred_df must have the same index and columns.")
    if per == 'cell_state':
        index, data_y, data_y_pred = signal_df.columns, signal_df.values.T, pred_df.values.T
    elif per == 'region':
        index, data_y, data_y_pred = signal_df.index, signal_df.values, pred_df.values
    else:
        raise ValueError(f"per must be 'cell_state' or 'region', not {per!r}.")

    pearson_r = numpy.full(len(index), numpy.nan)
    spearman_r = numpy.full(len(index), numpy.nan)
    rmse = numpy.full(len(index), numpy.nan)
    n_valid = numpy.zeros(len(index), dtype=int)
    for i, (y, y_pred) in enumerate(zip(data_y, data_y_pred)):
        valid = numpy.isfinite(y) & numpy.isfinite(y_pred)
        y, y_pred = y[valid], y_pred[valid]
        n_valid[i] = len(y)
        if len(y) < 1:
            continue
        rmse[i] = numpy.sqrt(numpy.mean((y - y_pred) ** 2))
        if len(y) < 2 or (y == y[0]).all() or (y_pred == y_pred[0]).all():
            continue
        pearson_r[i] = scipy.stats.pearsonr(y, y_pred)[0]
        spearman_r[i] = scipy.stats.spearmanr(y, y_pred)[0]

    return pandas.DataFrame(
        {'pearson_r': pearson_r, 'spearman_r': spearman_r, 'rmse': rmse, 'n_valid': n_valid},
        index=index,
    )
