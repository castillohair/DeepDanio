"""
Select cell state-specific peaks for computing contribution scores.

Peaks whose ensemble predictions have a mean squared error (MSE) of
MSE_THRESHOLD or more against measured accessibility are discarded. In each
cell state, the remaining peaks are ranked by predicted specificity, i.e. the
prediction in that cell state minus the mean prediction in all others, and the
top N_PEAKS_PER_CELL_STATE are selected. A peak can be selected in several
cell states.

Outputs:
- Ensemble predictions of all peaks, with datasets 'peak_id' and 'pred'
  (n_peaks, n_cell_states). Reused if present.
  Predicting takes about 25 minutes on a g5.xlarge instance.
- The selected peaks table, with columns 'cell_state', 'peak_id', 'rank'
  (0-based), 'specificity', and 'mse'.

Reruns can give slightly different predictions than the original run, which
can change the order of peaks with nearly equal specificity and the peaks
selected near the MSE and rank cutoffs. Disabling TensorFloat-32 (TF32), which
TensorFlow uses by default on recent GPUs, brings predictions closer to the
original ones. The released table is the original selection.

"""
import h5py
import numpy
import pandas
import tensorflow

from deepdanio import data, definitions, model, sequence

MSE_THRESHOLD = 0.125
N_PEAKS_PER_CELL_STATE = 10000

if __name__ == '__main__':
    # Full float32 precision, as in the original predictions
    tensorflow.config.experimental.enable_tensor_float_32_execution(False)

    peaks_df, signal_df = data.load_training_data('peaks')

    # Ensemble predictions on all peaks
    if definitions.ENSEMBLE_PREDICTIONS_PATH.exists():
        print(f"Loading predictions from {definitions.ENSEMBLE_PREDICTIONS_PATH}...")
        with h5py.File(definitions.ENSEMBLE_PREDICTIONS_PATH, 'r') as f:
            if not numpy.array_equal(f['peak_id'].asstr()[:], peaks_df.index.values):
                raise ValueError("Peaks in the predictions file differ from the training data.")
            pred = f['pred'][:]
    else:
        print(f"Predicting {len(peaks_df):,} peaks...")
        ensemble = model.make_model_ensemble([str(p) for p in definitions.DEEPDANIO_MODEL_PATHS.values()])
        pred = ensemble.predict(sequence.one_hot_encode(peaks_df['sequence'].values), batch_size=256, verbose=2)
        definitions.ENSEMBLE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(definitions.ENSEMBLE_PREDICTIONS_PATH, 'w') as f:
            f.create_dataset('peak_id', data=peaks_df.index.tolist(), dtype=h5py.string_dtype())
            f.create_dataset('pred', data=pred, compression='gzip')
        print(f"Predictions saved to {definitions.ENSEMBLE_PREDICTIONS_PATH}.")

    # Discard poorly predicted peaks
    pred_df = pandas.DataFrame(pred.astype('float64'), index=peaks_df.index, columns=definitions.CELL_STATES)
    mse = ((pred_df - signal_df.astype('float64')) ** 2).mean(axis=1)
    pred_df = pred_df[mse < MSE_THRESHOLD]
    print(f"{len(pred_df):,} / {len(peaks_df):,} peaks with MSE < {MSE_THRESHOLD}.")

    # Top peaks by predicted specificity in each cell state
    selected_dfs = []
    pred_sum = pred_df.sum(axis=1)
    n_other = len(definitions.CELL_STATES) - 1
    for cell_state in definitions.CELL_STATES:
        specificity = pred_df[cell_state] - (pred_sum - pred_df[cell_state]) / n_other
        specificity = specificity.sort_values(ascending=False).iloc[:N_PEAKS_PER_CELL_STATE]
        selected_dfs.append(pandas.DataFrame({
            'cell_state': cell_state,
            'peak_id': specificity.index,
            'rank': numpy.arange(len(specificity)),
            'specificity': specificity.values,
            'mse': mse[specificity.index].values,
        }))
    selected_df = pandas.concat(selected_dfs, ignore_index=True)

    definitions.SELECTED_PEAKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(definitions.SELECTED_PEAKS_PATH, sep='\t', index=False)
    print(
        f"Selected {len(selected_df):,} peaks ({selected_df['peak_id'].nunique():,} unique), "
        f"saved to {definitions.SELECTED_PEAKS_PATH}."
    )
