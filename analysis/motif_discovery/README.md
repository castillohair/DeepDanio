# Motif discovery

This folder computes contribution scores of cell state-specific peaks and discovers motifs from them:

1. Peak selection: the top 10,000 cell state-specific peaks of each of the 95 cell states are selected (162,262 unique peaks).
2. Contribution scores: contributions of each base to the predicted accessibility of each cell state are computed with DeepSHAP on all selected peaks in all cell states, on the ensemble of the models trained on splits 0, 1, and 2, using dinucleotide-shuffled reference sequences.
3. Motif discovery: in each cell state, TF-MoDISco finds motifs in the hypothetical contributions of the cell state's selected peaks. Motifs are trimmed, and their seqlets are mapped to positions in the peaks.
4. Motif clustering: the top 10 motifs of each cell state by number of seqlets are clustered across cell states by PWM similarity. Clusters were named manually, by similarity to known transcription factor motifs.

Computing all contributions takes about 1,000 GPU-hours. We therefore provide the precomputed results, which the scripts in this folder reproduce.

## Downloading results

`download_results.py` downloads the following files:

- `data/predictions/ensemble_predictions.h5` (180 MB): ensemble predictions of all peaks, with datasets `peak_id` and `pred` with shape (444,653, 95). Only needed to rerun peak selection without a GPU.
- `data/contributions/selected_peaks.tsv`: the selected peaks of each cell state, with columns `cell_state`, `peak_id`, `rank` (0-based), `specificity`, and `mse`. A peak can be selected in several cell states.
- `data/contributions/contributions.h5` (26 GB): actual contributions, i.e. contributions of the bases present in each sequence. Datasets are `peak_id`, `cell_state`, and `contributions` with shape (162,262, 95, 500). Peaks are stored one per chunk, so that single peaks can be read quickly.
- Into `data/motifs/`, motifs, seqlets, motif clusters, and cluster names (110 MB):
  - `00/` to `94/`: one folder per cell state, named by its index in `definitions.CELL_STATES`, with trimmed CWMs and PWMs (`cwm_trimmed.meme`, `pwm_trimmed.meme`), and a table of seqlets (`seqlets.tsv`). Motif IDs have the format `{pos/neg}_patterns_pattern_{idx}`, with patterns numbered by decreasing number of seqlets. The seqlets table has columns `motif`, `peak_id`, `start`, `end` (0-based, exclusive), `revcomp`, `cwm_contrib`, `cwm_match`, and `pwm_prob`.
  - `clustering/`: matrix-clustering inputs and results, the cluster of each motif (`motif_clusters.tsv`), and the consensus PWM of each cluster (`cluster_pwms.meme`).
  - `motif_cluster_names.tsv`: manually assigned name of each motif cluster. Names apply only to the released clustering, since rerunning matrix-clustering with different software versions or settings can change cluster numbering and membership.
- Into `data/motifs/`, TF-MoDISco results (1.1 GB): `modisco_results.h5` in each cell state's folder. Only needed to rerun motif extraction without TF-MoDISco.

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
python analysis/motif_discovery/select_peaks.py
```

Peaks whose ensemble predictions have a mean squared error of 0.125 or more against measured accessibility are discarded. In each cell state, the remaining peaks are ranked by predicted specificity, i.e. the prediction in that cell state minus the mean prediction in all others, and the top 10,000 are selected.

The script first predicts all peaks and saves the predictions to `data/predictions/ensemble_predictions.h5`, which is reused in later runs or can be downloaded instead. Predicting takes about 25 minutes on an NVIDIA A10G GPU. It then saves the selected peaks to `data/contributions/selected_peaks.tsv`.

Reruns can give slightly different predictions than the original run, which can change the order of peaks with nearly equal specificity and the peaks selected near the MSE and rank cutoffs. The released table is the original selection.

### Hypothetical contributions

```
usage: compute_contributions.py [-h] (--cell-state CELL_STATE | --combine)
                                [--cell-state-peaks-only | --peak-ids PEAK_IDS]
                                [--hypothetical-dir HYPOTHETICAL_DIR]
                                [--output OUTPUT]
```

With `--cell-state`, hypothetical contributions of all selected peaks are computed in one cell state, given by name or by index in `definitions.CELL_STATES`. For example, for adaxial cells (6-somite):

```shell
python analysis/motif_discovery/compute_contributions.py --cell-state 91
```

This produces `data/contributions/hypothetical/91.h5`, with datasets `peak_id` and `contributions` with shape (162,262, 500, 4). Each cell state takes about 9 hours on an NVIDIA A10G GPU, so we recommend running cell states in parallel on separate GPUs. Progress is saved every 1,000 peaks to a `.partial` file, from which interrupted runs resume.

Options:

- `--cell-state-peaks-only`: only compute contributions of the 10,000 peaks selected in the cell state, which takes about 35 minutes. This is enough for motif discovery.
- `--peak-ids`: compute contributions of any peaks in the training data, listed one per line in a text file. Requires `--hypothetical-dir`, to avoid mixing these results with those of the selected peaks.

Recomputed contributions differ from the released ones by about 1e-6 on average, and GPU and CPU results by about 1e-8.

### Combining contributions

```shell
python analysis/motif_discovery/compute_contributions.py --combine
```

This reads the hypothetical contributions of all 95 cell states, which must contain the same peaks, and writes the actual contributions to `data/contributions/contributions.h5`.

### Motif discovery

```
usage: find_motifs.py [-h] --cell-state CELL_STATE [--motifs-dir MOTIFS_DIR]
```

Runs TF-MoDISco on one cell state, given by name or index, and extracts trimmed motifs and seqlets. For example, for adaxial cells (6-somite):

```shell
python analysis/motif_discovery/find_motifs.py --cell-state 91
```

This requires the selected peaks. TF-MoDISco is skipped if `modisco_results.h5` is already present in the cell state's folder. Otherwise, it also requires the cell state's hypothetical contributions, which only need to include the cell state's own selected peaks (`compute_contributions.py --cell-state-peaks-only`).

TF-MoDISco takes 30 to 60 minutes per cell state with 4 threads on an Intel Xeon Platinum CPU, and needs about 2 GB of memory. Motif extraction runs on a single CPU core and takes a few minutes. Cell states are independent, so we recommend processing them in parallel.

Rerunning the script reproduces the released results.

### Motif clustering

```
usage: cluster_motifs.py [-h] [--motifs-dir MOTIFS_DIR]
                         [--clustering-dir CLUSTERING_DIR]
                         [--matrix-clustering MATRIX_CLUSTERING]
                         [--n-threads N_THREADS]
```

Selects the top 10 motifs of each cell state and clusters them with [matrix-clustering](https://github.com/jaimicore/matrix-clustering_stand-alone) from RSAT, using complete linkage and a normalized correlation threshold of 0.55. Results are saved in `data/motifs/clustering/`.

matrix-clustering requires R. The released clustering can be reproduced exactly with R 4.5.3 and matrix-clustering commit `a749c3f`; commits before February 22, 2024 give different clusters. To install it:

```shell
git clone https://github.com/jaimicore/matrix-clustering_stand-alone
cd matrix-clustering_stand-alone
git checkout a749c3f
cd compare-matrices-quick
make
```

Then install the required R packages from R:

```R
install.packages(c(
    "dplyr", "data.table", "furrr", "optparse", "purrr", "rcartocolor", "reshape2", "this.path", "tidyr",
    "dendsort", "ggplot2", "ggseqlogo", "RColorBrewer", "ape", "RJSONIO", "rjson", "circlize", "flexclust",
    "htmlwidgets", "plotly", "svglite", "jsonlite", "BiocManager"
))
BiocManager::install(c("universalmotif", "ComplexHeatmap"))
```

Depending on the Linux environment, some R packages may require the following system libraries to compile: libcurl, OpenSSL, libuv, fontconfig, FreeType, HarfBuzz, FriBidi, and libpng. On Ubuntu, [Posit Package Manager](https://packagemanager.posit.co/) provides precompiled R packages, which install much faster. Then run:

```shell
python analysis/motif_discovery/cluster_motifs.py --matrix-clustering path/to/matrix-clustering_stand-alone/matrix-clustering.R
```

matrix-clustering must be run with one thread (the default `--n-threads 1`), as it fails with more threads with current R package versions. It takes about 15 minutes. Without `--matrix-clustering`, the script writes the matrix-clustering inputs and prints the command to run it, so that clustering can be done on another machine. Once the results are copied to `data/motifs/clustering/`, rerunning the script processes them.
