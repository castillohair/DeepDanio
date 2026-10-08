"""
Compute contribution scores of selected peaks.

With --cell-state, hypothetical contributions are computed in one cell state
with DeepSHAP, on the ensemble of the models trained on all chromosome splits
and using dinucleotide-shuffled references. By default, all selected peaks are
used, i.e. peaks selected in any cell state. Alternatively, contributions can
be restricted to the peaks selected in the cell state being interpreted, which
are the input for motif discovery, or computed on a list of peaks from the
training data, selected or not. The output is one file per cell state in the
hypothetical contributions directory, with datasets 'peak_id' and
'contributions' with shape (n_peaks, seq_length, 4), and attribute
'cell_state'. Peaks are in the order of the training data.
Results are saved every BLOCK_SIZE peaks to a temporary file, from which
interrupted runs resume, and which is renamed when complete.

With --combine, hypothetical contributions of all cell states, which must
contain the same peaks, are combined into the actual contributions file:
contributions of the bases present in each sequence, with datasets 'peak_id',
'cell_state', and 'contributions' with shape (n_peaks, n_cell_states,
seq_length).

TF32 is disabled on GPUs that support it, for full float32 precision as in the
original contributions. Recomputed contributions match the original ones to
about 1e-5.

"""
import argparse
import contextlib
from pathlib import Path

import h5py
import numpy
import pandas
import tensorflow

from deepdanio import data, definitions, interpret, model, sequence

# Peaks computed or combined at a time
BLOCK_SIZE = 1000

# Output datasets have one chunk per peak, for fast access to single peaks
DATASET_KWARGS = dict(dtype='float32', compression='gzip', compression_opts=4, shuffle=True)


def compute_cell_state(cell_state, hypothetical_dir, cell_state_peaks_only=False, peak_ids_path=None):
    """
    Compute hypothetical contributions of peaks in one cell state.

    Parameters
    ----------
    cell_state : str
        Cell state name, or index as a string.
    hypothetical_dir : pathlib.Path
        Output directory.
    cell_state_peaks_only : bool, optional
        Whether to only use peaks selected in this cell state, instead of
        peaks selected in any cell state.
    peak_ids_path : pathlib.Path, optional
        Text file with one peak ID per line, from the training data. If
        given, these peaks are used instead of the selected ones.

    """
    cell_state_idx = int(cell_state) if cell_state.isdigit() else definitions.CELL_STATES.index(cell_state)
    cell_state = definitions.CELL_STATES[cell_state_idx]
    output_path = hypothetical_dir / definitions.HYPOTHETICAL_CONTRIBUTIONS_NAME.format(cell_state_idx=cell_state_idx)
    partial_path = output_path.with_name(output_path.name + '.partial')
    print(f"Cell state {cell_state_idx}: {cell_state}")
    print(f"Output: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists.")

    # Peaks to interpret, in training data order
    peaks_df, _ = data.load_training_data('peaks')
    if peak_ids_path is not None:
        peak_ids = set(peak_ids_path.read_text().split())
        unknown_peak_ids = peak_ids - set(peaks_df.index)
        if unknown_peak_ids:
            raise ValueError(
                f"{len(unknown_peak_ids)} requested peaks are not in the training data, e.g. {unknown_peak_ids.pop()}."
            )
    else:
        selected_df = pandas.read_csv(definitions.SELECTED_PEAKS_PATH, sep='\t')
        if cell_state_peaks_only:
            selected_df = selected_df[selected_df['cell_state'] == cell_state]
        peak_ids = set(selected_df['peak_id'])
    peaks_df = peaks_df[peaks_df.index.isin(peak_ids)]
    n_peaks = len(peaks_df)

    # Create the temporary output file, or resume from it
    hypothetical_dir.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        with h5py.File(partial_path, 'r') as f:
            if f.attrs['cell_state'] != cell_state or not numpy.array_equal(f['peak_id'].asstr()[:], peaks_df.index):
                raise ValueError(f"{partial_path} was created for different peaks or cell state.")
            n_computed = f.attrs['n_computed']
        print(f"Resuming from {partial_path}, with {n_computed:,} / {n_peaks:,} peaks computed.")
    else:
        with h5py.File(partial_path, 'w') as f:
            f.attrs['cell_state'] = cell_state
            f.attrs['n_computed'] = 0
            f.create_dataset('peak_id', data=peaks_df.index.tolist(), dtype=h5py.string_dtype())
            f.create_dataset(
                'contributions',
                shape=(n_peaks, definitions.SEQ_LENGTH, 4),
                chunks=(1, definitions.SEQ_LENGTH, 4),
                **DATASET_KWARGS,
            )
        n_computed = 0

    print(f"Computing contributions of {n_peaks:,} peaks...")
    ensemble = model.make_model_ensemble([str(p) for p in definitions.DEEPDANIO_MODEL_PATHS.values()])
    for block_start in range(n_computed, n_peaks, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_peaks)
        seqs_onehot = sequence.one_hot_encode(peaks_df['sequence'].values[block_start:block_end])
        contribs = interpret.compute_contributions(ensemble, seqs_onehot, cell_state_idx)
        with h5py.File(partial_path, 'a') as f:
            f['contributions'][block_start:block_end] = contribs
            f.attrs['n_computed'] = block_end
        print(f"Computed {block_end:,} / {n_peaks:,} peaks.", flush=True)

    with h5py.File(partial_path, 'a') as f:
        del f.attrs['n_computed']
    partial_path.rename(output_path)
    print(f"Contributions saved to {output_path}.")


def combine(hypothetical_dir, output_path):
    """
    Combine hypothetical contributions of all cell states into actual contributions.

    Parameters
    ----------
    hypothetical_dir : pathlib.Path
        Directory with one hypothetical contributions file per cell state.
    output_path : pathlib.Path
        Output file.

    """
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists.")
    hyp_paths = [
        hypothetical_dir / definitions.HYPOTHETICAL_CONTRIBUTIONS_NAME.format(cell_state_idx=cell_state_idx)
        for cell_state_idx in range(len(definitions.CELL_STATES))
    ]
    missing_paths = [p for p in hyp_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} hypothetical contributions files missing, e.g. {missing_paths[0]}.")

    with contextlib.ExitStack() as stack:
        hyp_files = [stack.enter_context(h5py.File(p, 'r')) for p in hyp_paths]
        peak_ids = hyp_files[0]['peak_id'].asstr()[:]
        for hyp_path, hyp_file in zip(hyp_paths, hyp_files):
            if not numpy.array_equal(hyp_file['peak_id'].asstr()[:], peak_ids):
                raise ValueError(f"Peaks in {hyp_path} differ from those in {hyp_paths[0]}.")
        peaks_df, _ = data.load_training_data('peaks')
        seqs_onehot = sequence.one_hot_encode(peaks_df.loc[peak_ids, 'sequence'].values)
        n_peaks = len(peak_ids)
        n_cell_states = len(definitions.CELL_STATES)
        print(f"Combining contributions of {n_peaks:,} peaks in {n_cell_states} cell states...")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('peak_id', data=peak_ids.tolist(), dtype=h5py.string_dtype())
            f.create_dataset('cell_state', data=definitions.CELL_STATES, dtype=h5py.string_dtype())
            contrib_dataset = f.create_dataset(
                'contributions',
                shape=(n_peaks, n_cell_states, definitions.SEQ_LENGTH),
                chunks=(1, n_cell_states, definitions.SEQ_LENGTH),
                **DATASET_KWARGS,
            )
            for block_start in range(0, n_peaks, BLOCK_SIZE):
                block_end = min(block_start + BLOCK_SIZE, n_peaks)
                # Hypothetical contributions with shape (n_block_peaks, n_cell_states, seq_length, 4)
                hyp_block = numpy.stack(
                    [hyp_file['contributions'][block_start:block_end] for hyp_file in hyp_files],
                    axis=1,
                )
                contrib_dataset[block_start:block_end] = (hyp_block * seqs_onehot[block_start:block_end, None]).sum(axis=-1)
                print(f"Combined {block_end:,} / {n_peaks:,} peaks.", flush=True)

    print(f"Contributions saved to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cell-state', type=str, help='Cell state name or index to compute contributions in.')
    mode.add_argument('--combine', action='store_true', help='Combine contributions of all cell states.')
    peak_source = parser.add_mutually_exclusive_group()
    peak_source.add_argument(
        '--cell-state-peaks-only',
        action='store_true',
        help='Only use peaks selected in the cell state, instead of peaks selected in any cell state.',
    )
    peak_source.add_argument(
        '--peak-ids',
        type=Path,
        help='Text file with one peak ID per line, from the training data, to use instead of the selected peaks. '
        'Requires --hypothetical-dir.',
    )
    parser.add_argument(
        '--hypothetical-dir',
        type=Path,
        help=f'Directory with one hypothetical contributions file per cell state. '
        f'Default: {definitions.HYPOTHETICAL_CONTRIBUTIONS_DIR}.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=definitions.CONTRIBUTIONS_PATH,
        help='Actual contributions file written with --combine.',
    )
    args = parser.parse_args()
    # Keep contributions of custom peak lists out of the default directory
    if args.peak_ids is not None and args.hypothetical_dir is None:
        parser.error('--peak-ids requires --hypothetical-dir.')
    if args.hypothetical_dir is None:
        args.hypothetical_dir = definitions.HYPOTHETICAL_CONTRIBUTIONS_DIR

    if args.combine:
        combine(args.hypothetical_dir, args.output)
    else:
        tensorflow.config.experimental.enable_tensor_float_32_execution(False)
        compute_cell_state(args.cell_state, args.hypothetical_dir, args.cell_state_peaks_only, args.peak_ids)
