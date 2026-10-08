# Contribution scores

Contribution scores of each base to the predicted accessibility of each cell state are computed with DeepSHAP, on the ensemble of the models trained on splits 0, 1, and 2, using dinucleotide-shuffled reference sequences. They are computed on the top 10,000 cell state-specific peaks of each of the 95 cell states (162,262 unique peaks), in all cell states.

Computing all contributions takes about 1,000 GPU-hours. We therefore provide the precomputed results, which the scripts in this folder reproduce.

## Downloading results

`download_contributions.py` downloads the following files:

- `data/predictions/ensemble_predictions.h5` (180 MB): ensemble predictions of all peaks, with datasets `peak_id` and `pred` with shape (444,653, 95). Only needed to rerun peak selection without a GPU.
- `data/contributions/selected_peaks.tsv`: the selected peaks of each cell state, with columns `cell_state`, `peak_id`, `rank` (0-based), `specificity`, and `mse`. A peak can be selected in several cell states.
- `data/contributions/contributions.h5` (26 GB): actual contributions, i.e. contributions of the bases present in each sequence. Datasets are `peak_id`, `cell_state`, and `contributions` with shape (162,262, 95, 500). Peaks are stored one per chunk, so that single peaks can be read quickly.

Hypothetical contributions, i.e. contributions of all four bases at each position, are available upon request, or can be recomputed with `compute_contributions.py`.

To load the contributions of some peaks in some cell states:

```python
from deepdanio import data

contribs = data.load_contributions(peak_ids, cell_states)  # shape (n_peaks, n_cell_states, 500)
```

See [`examples/contributions.ipynb`](../../examples/contributions.ipynb) for loading and plotting the contributions of one peak.

## Reproducing results

Scripts are run in the following order, from the repository root. They require the processed training data (see `data/download_data.py`) and the pretrained model weights (see `models/download_weights.py`).

### Peak selection

```shell
python analysis/contributions/select_peaks.py
```

Peaks whose ensemble predictions have a mean squared error of 0.125 or more against measured accessibility are discarded. In each cell state, the remaining peaks are ranked by predicted specificity, i.e. the prediction in that cell state minus the mean prediction in all others, and the top 10,000 are selected.

The script first predicts all peaks and saves the predictions to `data/predictions/ensemble_predictions.h5`, which is reused in later runs or can be downloaded instead. Predicting takes about 25 minutes on an NVIDIA A10G GPU (AWS g5.xlarge instance). It then saves the selected peaks to `data/contributions/selected_peaks.tsv`.

Small floating point differences across hardware can swap the order of peaks with nearly equal specificity. The released table is the original selection.

### Hypothetical contributions

```
usage: compute_contributions.py [-h] (--cell-state CELL_STATE | --combine)
                                [--cell-state-peaks-only | --peak-ids PEAK_IDS]
                                [--hypothetical-dir HYPOTHETICAL_DIR]
                                [--output OUTPUT]
```

With `--cell-state`, hypothetical contributions of all selected peaks are computed in one cell state, given by name or by index in `definitions.CELL_STATES`. For example, for adaxial cells (6-somite):

```shell
python analysis/contributions/compute_contributions.py --cell-state 91
```

This produces `data/contributions/hypothetical/91.h5`, with datasets `peak_id` and `contributions` with shape (162,262, 500, 4). Each cell state takes about 9 hours on an NVIDIA A10G GPU (AWS g5.xlarge instance), so we recommend running cell states in parallel on separate GPUs. Progress is saved every 1,000 peaks to a `.partial` file, from which interrupted runs resume.

Options:

- `--cell-state-peaks-only`: only compute contributions of the 10,000 peaks selected in the cell state, which takes about 35 minutes.
- `--peak-ids`: compute contributions of any peaks in the training data, listed one per line in a text file. Requires `--hypothetical-dir`, to avoid mixing these results with those of the selected peaks.

Recomputed contributions differ from the released ones by about 1e-6 on average, and GPU and CPU results by about 1e-8.

### Combining contributions

```shell
python analysis/contributions/compute_contributions.py --combine
```

This reads the hypothetical contributions of all 95 cell states, which must contain the same peaks, and writes the actual contributions to `data/contributions/contributions.h5`.
