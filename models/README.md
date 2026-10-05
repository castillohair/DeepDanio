# Models

## Downloading model weights

`download_weights.py` downloads the final weights of each model into its folder (e.g. `deepdanio/deepdanio_split_0.h5`).

## DeepDanio

`deepdanio/train.py` trains a model on one chromosome split, using the processed data in `data/processed`. Usage:

```
usage: train.py [-h] [--data-split-idx DATA_SPLIT_IDX] [--n-rounds N_ROUNDS]
                [--output-name OUTPUT_NAME] [--seed SEED] [--save-intermediate]
                [--save-checkpoints]
```

Training proceeds in rounds. Each round trains until validation loss stops improving, keeping the best weights, and starts with a new optimizer. Models were trained with two rounds (the default) on chromosome splits 0, 1, and 2. For split 0:

```shell
python deepdanio/train.py --data-split-idx 0
```

This produces `deepdanio/deepdanio_split_0.h5`. Existing models are not overwritten.
