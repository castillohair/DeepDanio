# DeepDanio

**Predict chromatin accessibility across 95 zebrafish embryonic cell states.** DeepDanio is a deep learning model trained on pseudobulk single-cell ATAC-seq data from zebrafish embryogenesis, covering cell states from the high stage to the 6-somite stage. This repository contains code associated with our [preprint](https://doi.org/10.1101/2024.08.27.609971). It can be used to make predictions with pretrained models, reproduce model training, evaluate model performance, and analyze contribution scores and sequence motifs.

The list of cell states predicted by DeepDanio, along with their cell type, developmental stage, and lineage, can be found [here](./src/deepdanio/resources/cell_state_metadata.csv).

## Contents

- **[`src/deepdanio`](./src/deepdanio/)**: Python package with code used across the repository.
  - `definitions.py`: paths to data and models, sequence length, and cell state information.
  - `data.py`, `genome.py`, `sequence.py`: loading of processed data and the reference genome, and sequence manipulation.
  - `model.py`: model architecture, loading of trained models, and model ensembles.
  - `metrics.py`: prediction performance metrics.
  - `interpret.py`, `motif.py`: contribution scores (DeepSHAP), motif scanning, and motif file handling.
  - `plot.py`: plots of values across cell states, grouped by stage or along the differentiation trajectory, and sequence and contribution logos.
  - `resources/`: cell state metadata, and differentiation trajectory layout and paths.
- **[`examples`](./examples/)**: notebooks showing how to use DeepDanio.
  - `predict.ipynb`: predict accessibility of a genomic region or a custom sequence, and plot predictions.
  - `contributions.ipynb`: load and plot contribution scores of a cell state-specific peak.
- **[`data`](./data/)**: data download and processing.
  - `download_data.py`: downloads the reference genome, raw scATAC-seq data, and processed training data.
  - `process.py`: generates training data (peak sequences, normalized signal, and random genomic negative regions) and chromosome-based data splits from raw data.
  - `processed/chr_splits.json`: train/validation/test chromosomes of each data split.
- **[`models`](./models/)**: model training and pretrained weights. See the folder's [README](./models/README.md) for more information.
  - `download_weights.py`: downloads pretrained weights.
  - `deepdanio/train.py`: trains a model on one data split.
- **[`analysis`](./analysis/)**: analyses of trained models.
  - `model_performance/`: prediction performance on held-out chromosomes.
  - `contributions/`: selection of cell state-specific peaks and calculation of their contribution scores (DeepSHAP). See the folder's [README](./analysis/contributions/README.md) for more information.
  - `motifs/`: motif discovery (TF-MoDISco), clustering across cell states, and motif contributions along differentiation paths. See the folder's [README](./analysis/motifs/README.md) for more information.

## Models

DeepDanio is a dilated residual convolutional network that takes a 500 bp sequence and outputs one accessibility value per cell state. Outputs are log10-transformed CPM, quantile normalized across cell states.

Data are split by chromosome into 10 groups with similar numbers of peaks. In split *i*, group *i* is used for testing and group *i* + 1 for validation. We provide models trained on splits 0, 1, and 2, whose test chromosomes do not overlap. We recommend using the ensemble of all three models for general predictions, and individual models when evaluating on native sequences from their test chromosomes.

## Use cases

### Predicting accessibility

Install the [requirements](#installation-guide) and download pretrained weights:

```
cd models
python download_weights.py
cd ..
```

Then follow [`examples/predict.ipynb`](./examples/predict.ipynb) to predict accessibility of a genomic region or a sequence of your own, and plot predictions. Predicting on genomic regions also requires the GRCz11 reference genome, which can be downloaded via `data/download_data.py`.

Briefly, predictions can be obtained as follows:

```python
from deepdanio import definitions, model, sequence

model_paths = [str(p) for p in definitions.DEEPDANIO_MODEL_PATHS.values()]
keras_model = model.make_model_ensemble(model_paths)
pred = keras_model.predict(sequence.one_hot_encode([seq]))  # shape (1, 95)
```

Output columns follow the order of `definitions.CELL_STATES`.

### Reproducing model training

Download the processed training data via `data/download_data.py`. Alternatively, download the reference genome and raw scATAC-seq data with the same script and regenerate the processed data via `data/process.py`.

Then train one model per split, e.g. for split 0:

```
python models/deepdanio/train.py --data-split-idx 0
```

See the [`models`](./models/) folder's [README](./models/README.md) for more information.

### Evaluating model performance

Run `analysis/model_performance/predict.py` to predict the validation and test chromosomes of each split with its corresponding model, then follow [`analysis/model_performance/analysis.ipynb`](./analysis/model_performance/analysis.ipynb) to calculate performance metrics per cell state and per peak.

### Exploring contribution scores

Download the precomputed contribution scores and the processed training data:

```
python analysis/contributions/download_contributions.py
python data/download_data.py
```

Then follow [`examples/contributions.ipynb`](./examples/contributions.ipynb) to load and plot the contributions of a peak. Briefly:

```python
from deepdanio import data

contribs = data.load_contributions(peak_ids, cell_states)  # shape (n_peaks, n_cell_states, 500)
```

### Computing contributions of new sequences

Contribution scores of any sequence in one cell state can be computed with DeepSHAP on the model ensemble:

```python
from deepdanio import definitions, interpret, model, sequence

model_paths = [str(p) for p in definitions.DEEPDANIO_MODEL_PATHS.values()]
keras_model = model.make_model_ensemble(model_paths)
seqs_onehot = sequence.one_hot_encode(seqs)
hyp_contribs = interpret.compute_contributions(keras_model, seqs_onehot, definitions.CELL_STATES.index(cell_state))
contribs = (hyp_contribs * seqs_onehot).sum(axis=-1)  # shape (n_seqs, 500)
```

`hyp_contribs` are hypothetical contributions, i.e. the contributions each of the four bases would have at each position, with shape (n_seqs, 500, 4). Computing contributions takes about 0.2 seconds per sequence and cell state on a GPU, and about 17 seconds on 4 CPU cores, so only a small number of sequences can be run on a CPU.

### Reproducing contribution and motif analyses

See [`analysis/contributions`](./analysis/contributions/) and then [`analysis/motifs`](./analysis/motifs/). Precomputed results can be downloaded with `download_contributions.py` and `download_motifs.py` in those folders.

## Requirements

### Hardware requirements

Model training and computing contribution scores at scale require an NVIDIA GPU. Predictions and contributions of a small number of sequences can be computed on a regular CPU. Motif discovery runs on CPU, and motif clustering requires R (see [`analysis/motifs`](./analysis/motifs/)).

### Software requirements

#### OS requirements

Tested on macOS (Apple Silicon) and Linux. On Linux x86_64, the CUDA runtime required by TensorFlow is installed as a Python dependency, so only the NVIDIA driver needs to be installed on the host.

#### Python dependencies

This repo requires Python 3.11. Dependencies are declared in `pyproject.toml` and pinned to exact versions in `uv.lock`. They include:

- Standard packages: `numpy` (1.x), `scipy`, `pandas`, `h5py`, `matplotlib`, `seaborn`, `biopython`, `logomaker`, `prtpy`, `numba`, `memelite`, `modisco-lite`, `leidenalg`. `leidenalg` is pinned to 0.11 to reproduce the released motifs.
- `tensorflow` 2.14 with Keras 2. Note that starting with `tensorflow` 2.16, Keras 3 is included by default and may not work out of the box with this repository.
- A [modified version of SHAP](https://github.com/castillohair/shap) for DeepSHAP contribution scores on genomic sequences, which requires `deeplift`. It is installed from GitHub and compiled during installation.

## Installation guide

We recommend using [uv](https://docs.astral.sh/uv/). To download the repository and install all requirements run the following:

```
git clone https://github.com/castillohair/DeepDanio
cd DeepDanio
uv sync
source .venv/bin/activate
```

`uv sync` installs the exact versions recorded in `uv.lock`, and pulls the GPU-enabled dependencies only on Linux x86_64. It also installs this repository's shared code as the `deepdanio` package in editable mode.

Alternatively, install into an existing Python 3.11 environment by running the following from the repository root:

```
pip install -e .
```

Either method requires `git` and a C++ compiler, since the modified SHAP is downloaded from GitHub and compiled during installation. For example:

- macOS: `xcode-select --install`
- Ubuntu/Debian: `sudo apt install git build-essential`
- Amazon Linux/RHEL/Fedora: `sudo dnf install git gcc gcc-c++`

## Citation

If you use DeepDanio, please cite our [preprint](https://doi.org/10.1101/2024.08.27.609971).
