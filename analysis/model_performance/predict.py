"""
Predict held-out data with the model of each chromosome split.

For each split, the split's model predicts the peaks and negative regions on
the validation and test chromosomes of that split. Predictions are saved to
predictions.h5 in this folder, in groups "split_{i}/{subset}/{region_group}",
each with datasets:
- idx: row indices of the regions in the processed training data.
- pred: predictions, with shape (n_regions, n_cell_states).

"""
import argparse
import json
from pathlib import Path

import h5py
import numpy

from deepdanio import data, definitions, model, sequence

PREDICTIONS_PATH = Path(__file__).resolve().parent / 'predictions.h5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--splits',
        type=int,
        nargs='+',
        default=definitions.DEEPDANIO_SPLITS,
        help='Chromosome splits to predict.',
    )
    parser.add_argument(
        '--subsets',
        type=str,
        nargs='+',
        default=['val', 'test'],
        help='Subsets of each split to predict.',
    )
    args = parser.parse_args()

    print("Loading data...")
    regions = {}
    regions['peaks'], _ = data.load_training_data('peaks')
    regions['negatives'], _ = data.load_training_data('negatives')
    with open(definitions.CHR_SPLITS_PATH) as f:
        chr_splits = json.load(f)

    for split in args.splits:
        print(f"Loading model for split {split}...")
        keras_model = model.load_model(definitions.DEEPDANIO_MODEL_PATHS[split])

        for subset in args.subsets:
            for region_group, regions_df in regions.items():
                idx = numpy.flatnonzero(regions_df['chr'].isin(chr_splits[split][subset]).values)
                print(f"Predicting {len(idx):,} {region_group} in split {split}, {subset} set...")
                pred = keras_model.predict(
                    sequence.one_hot_encode(regions_df['sequence'].values[idx]),
                    batch_size=256,
                    verbose=0,
                )

                # Replace previous predictions of the same group, keep others
                group_name = f'split_{split}/{subset}/{region_group}'
                with h5py.File(PREDICTIONS_PATH, 'a') as f:
                    if group_name in f:
                        del f[group_name]
                    h5_group = f.create_group(group_name)
                    h5_group.create_dataset('idx', data=idx)
                    h5_group.create_dataset('pred', data=pred.astype('float32'), compression='gzip')

    print(f"Predictions saved to {PREDICTIONS_PATH}.")
