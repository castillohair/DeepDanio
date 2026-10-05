import argparse
import json

import numpy
import tensorflow

from deepdanio import data, definitions, model, sequence

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


class SeqGenerator(tensorflow.keras.utils.Sequence):
    """
    Generates batches of one-hot encoded sequences and signal targets.

    Each epoch, a random subset of negative sequences is added to the peaks.
    Negatives are assigned the minimum signal of each output across peaks.

    Parameters
    ----------
    seqs : numpy.ndarray
        Peak sequences.
    signal : numpy.ndarray
        Peak signal, with shape (n_peaks, n_outputs).
    negative_seqs : numpy.ndarray
        Pool of negative sequences.
    n_negatives_per_epoch : int, optional
        Number of negatives added each epoch. Default is 1/20 of the number
        of peaks.
    batch_size : int, optional
        Number of sequences per batch.
    shuffle : bool, optional
        Whether to shuffle sequences each epoch.

    """
    def __init__(
            self,
            seqs,
            signal,
            negative_seqs,
            n_negatives_per_epoch=None,
            batch_size=128,
            shuffle=True,
        ):
        self.seqs = seqs
        self.signal = signal
        self.negative_seqs = negative_seqs
        if n_negatives_per_epoch is None:
            n_negatives_per_epoch = int(len(seqs) / 20)
        self.n_negatives_per_epoch = n_negatives_per_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.negative_signal = numpy.tile(signal.min(axis=0), (n_negatives_per_epoch, 1))

        self.on_epoch_end()

    def __len__(self):
        return int(numpy.ceil(len(self.seqs_epoch) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return (
            sequence.one_hot_encode(self.seqs_epoch[batch_idx]),
            self.signal_epoch[batch_idx],
        )

    def on_epoch_end(self):
        """
        Resample negatives and shuffle.
        """
        negative_idx = numpy.random.choice(
            len(self.negative_seqs),
            size=self.n_negatives_per_epoch,
            replace=False,
        )
        self.seqs_epoch = numpy.concatenate([self.seqs, self.negative_seqs[negative_idx]])
        self.signal_epoch = numpy.concatenate([self.signal, self.negative_signal])

        self.indexes = numpy.arange(len(self.seqs_epoch))
        if self.shuffle:
            numpy.random.shuffle(self.indexes)


def train_model(
        data_split_idx=0,
        n_rounds=2,
        output_name=None,
        seed=None,
        save_intermediate=False,
        save_checkpoints=False,
    ):
    """
    Train a model on one data split.

    Training proceeds in rounds. Each round trains until validation loss
    stops improving, keeping the best weights, and starts with a new
    optimizer.

    Parameters
    ----------
    data_split_idx : int, optional
        Index of the chromosome split.
    n_rounds : int, optional
        Number of training rounds.
    output_name : str, optional
        Name of the output model, without extension. Default is the model name
        for this split.
    seed : int, optional
        Random seed for Python, NumPy, and TensorFlow.
    save_intermediate : bool, optional
        Whether to save the model after each round before the last, as
        "{output_name}_round_{i}".
    save_checkpoints : bool, optional
        Whether to save the model after each epoch.

    """
    if output_name is None:
        output_name = definitions.DEEPDANIO_MODEL_NAME.format(split=data_split_idx)
    output_path = definitions.DEEPDANIO_MODEL_DIR / f'{output_name}.h5'
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists.")
    print(f"Training model {output_name}...")

    if seed is not None:
        tensorflow.keras.utils.set_random_seed(seed)

    # Load data
    ###########
    print("Loading data...")
    peaks_df, signal_df = data.load_training_data('peaks')
    negatives_df, _ = data.load_training_data('negatives')
    print(f"{len(peaks_df):,} peaks and {len(negatives_df):,} negatives loaded.")

    with open(definitions.CHR_SPLITS_PATH) as f:
        chr_split = json.load(f)[data_split_idx]
    print(f"Using split {data_split_idx}.")
    print(f"Chromosomes for training: {chr_split['train']}")
    print(f"Chromosomes for validation: {chr_split['val']}")
    print(f"Chromosomes for testing: {chr_split['test']}")

    peaks_train = peaks_df['chr'].isin(chr_split['train']).values
    peaks_val = peaks_df['chr'].isin(chr_split['val']).values
    negatives_train = negatives_df['chr'].isin(chr_split['train']).values
    negatives_val = negatives_df['chr'].isin(chr_split['val']).values

    seqs_train = peaks_df['sequence'].values[peaks_train]
    signal_train = signal_df.values[peaks_train]
    negative_seqs_train = negatives_df['sequence'].values[negatives_train]
    seqs_val = peaks_df['sequence'].values[peaks_val]
    signal_val = signal_df.values[peaks_val]
    negative_seqs_val = negatives_df['sequence'].values[negatives_val]

    print(f"{len(seqs_train):,} peaks and {len(negative_seqs_train):,} negatives for training.")
    print(f"{len(seqs_val):,} peaks and {len(negative_seqs_val):,} negatives for validation.")

    # Add reverse complements to training peaks and negatives
    seqs_train = numpy.concatenate([seqs_train, [sequence.revcomp(s) for s in seqs_train]])
    signal_train = numpy.concatenate([signal_train, signal_train])
    negative_seqs_train = numpy.concatenate(
        [negative_seqs_train, [sequence.revcomp(s) for s in negative_seqs_train]]
    )

    # Model
    #######
    keras_model = model.make_resnet(
        definitions.SEQ_LENGTH,
        groups=4,
        blocks_per_group=3,
        filters=256,
        kernel_size=13,
        dilation_rates=[1, 2, 4, 8],
        first_conv_activation='relu',
        n_outputs=signal_df.shape[1],
        output_activation='linear',
    )

    # Train
    #######
    batch_size = 256
    generator_train = SeqGenerator(seqs_train, signal_train, negative_seqs_train, batch_size=batch_size)
    generator_val = SeqGenerator(seqs_val, signal_val, negative_seqs_val, batch_size=batch_size)

    for round_idx in range(1, n_rounds + 1):
        print(f"Training round {round_idx}/{n_rounds}...")
        round_name = output_name if round_idx == n_rounds else f'{output_name}_round_{round_idx}'

        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
            ),
        ]
        if save_checkpoints:
            callbacks.append(
                tensorflow.keras.callbacks.ModelCheckpoint(
                    filepath=str(definitions.DEEPDANIO_MODEL_DIR / (round_name + '_checkpoint_{epoch:02d}.h5')),
                )
            )

        # Compiling with a new optimizer resets its state
        keras_model.compile(
            loss=model.MSE_Cosine_Loss(w_mse=0.5, w_cosine=0.5),
            optimizer=tensorflow.keras.optimizers.Adam(2e-4),
        )
        keras_model.fit(
            x=generator_train,
            validation_data=generator_val,
            epochs=100,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )

        if round_idx == n_rounds or save_intermediate:
            round_path = definitions.DEEPDANIO_MODEL_DIR / f'{round_name}.h5'
            keras_model.save(round_path, include_optimizer=False)
            print(f"Model saved to {round_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-split-idx',
        type=int,
        default=0,
        help='Chromosome split index (0-9).',
    )
    parser.add_argument(
        '--n-rounds',
        type=int,
        default=2,
        help='Number of training rounds, each with a new optimizer.',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Output model name, without extension. Default is the model name for this split.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed.',
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save the model after each round before the last.',
    )
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        help='Save the model after each epoch.',
    )
    args = parser.parse_args()

    train_model(
        data_split_idx=args.data_split_idx,
        n_rounds=args.n_rounds,
        output_name=args.output_name,
        seed=args.seed,
        save_intermediate=args.save_intermediate,
        save_checkpoints=args.save_checkpoints,
    )
