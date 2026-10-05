# Models

## Downloading model weights

`download_weights.py` downloads the final weights of each model into its folder (e.g. `deepdanio/deepdanio_split_0.h5`).

## DeepDanio

`deepdanio/train.py` trains a model on one chromosome split, using the processed data in `data/processed`. Usage:

```
usage: train.py [-h] [--data-split-idx DATA_SPLIT_IDX] [--starting-model STARTING_MODEL]
                [--output-name OUTPUT_NAME] [--seed SEED]
```

Each model is trained in two stages: training until early stopping, then training again from the resulting model with a new optimizer. Models were trained on chromosome splits 0, 1, and 2. For split 0:

```shell
cd deepdanio
python train.py --data-split-idx 0
python train.py --data-split-idx 0 --starting-model deepdanio_split_0_intermediate.h5
```

This produces `deepdanio_split_0_intermediate.h5` and the final model `deepdanio_split_0.h5`. Use `--save-checkpoints` to also save the model after each epoch.
