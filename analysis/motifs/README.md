# Motifs

Motifs are discovered from the contribution scores of each cell state's selected peaks (see [`analysis/contributions`](../contributions/)), and then compared across cell states and differentiation paths:

1. In each cell state, TF-MoDISco finds motifs in the hypothetical contributions of the cell state's 10,000 selected peaks. Motifs are trimmed, and the selected peaks are scanned for matches using actual contributions.
2. The top 10 motifs of each cell state by number of seqlets are clustered across cell states by PWM similarity. Clusters were named manually, by similarity to known transcription factor motifs.
3. The contribution of each match to the top 5 positive motifs of each cell state is calculated in all cell states, and summarized along each of the 29 differentiation paths.

## Downloading results

`download_motifs.py` downloads the following into `data/motifs/`:

- Motifs, seqlets, matches, motif clusters, and cluster names (260 MB). These are enough to run `cluster_motifs.py` and `motif_contributions.ipynb`.
  - `00/` to `94/`: one folder per cell state, named by its index in `definitions.CELL_STATES`, with trimmed CWMs and PWMs (`cwm_trimmed.meme`, `pwm_trimmed.meme`), and tables of seqlets and matches (`seqlets.tsv`, `motif_matches.tsv`). Motif IDs have the format `{pos/neg}_patterns_pattern_{idx}`, with patterns numbered by decreasing number of seqlets. Tables have columns `motif`, `peak_id`, `start`, `end` (0-based, exclusive), `revcomp`, `cwm_contrib`, `cwm_match`, and `pwm_prob`.
  - `clustering/`: matrix-clustering inputs and results, the cluster of each motif (`motif_clusters.tsv`), and the consensus PWM of each cluster (`cluster_pwms.meme`).
  - `motif_cluster_names.tsv`: manually assigned name of each motif cluster. Names apply only to the released clustering, since rerunning matrix-clustering with different software versions or settings can change cluster numbering and membership.
- TF-MoDISco results (1.1 GB): `modisco_results.h5` in each cell state's folder. Only needed to rerun motif extraction and scanning without TF-MoDISco.

## Reproducing results

Scripts are run in the following order, from the repository root.

### Motif discovery and scanning

```
usage: find_motifs.py [-h] --cell-state CELL_STATE [--motifs-dir MOTIFS_DIR]
```

Runs TF-MoDISco on one cell state, given by name or index, extracts trimmed motifs and seqlets, and scans the cell state's selected peaks. For example, for adaxial cells (6-somite):

```shell
python analysis/motifs/find_motifs.py --cell-state 91
```

This requires the selected peaks and contributions (see [`analysis/contributions`](../contributions/)), and the processed training data. TF-MoDISco is skipped if `modisco_results.h5` is already present in the cell state's folder. Otherwise, it also requires the cell state's hypothetical contributions, which only need to include the cell state's own selected peaks (`compute_contributions.py --cell-state-peaks-only`).

Motif extraction and scanning run on a single CPU core and take about 15 to 30 minutes per cell state, with about 1.3 GB of memory. TF-MoDISco adds 30 to 60 minutes per cell state with 4 CPU threads on an AWS c5.4xlarge instance, and needs about 2 GB of memory. Cell states are independent, so we recommend processing them in parallel.

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
python analysis/motifs/cluster_motifs.py --matrix-clustering path/to/matrix-clustering_stand-alone/matrix-clustering.R
```

matrix-clustering must be run with one thread (the default `--n-threads 1`), as it fails with more threads with current R package versions. It takes about 15 minutes. Without `--matrix-clustering`, the script writes the matrix-clustering inputs and prints the command to run it, so that clustering can be done on another machine. Once the results are copied to `data/motifs/clustering/`, rerunning the script processes them.

### Motif contributions

[`motif_contributions.ipynb`](./motif_contributions.ipynb) calculates the contribution of each match to the top 5 positive motifs of each cell state in all cell states, and summarizes them per motif and per motif cluster along each differentiation path, as means and 90th percentiles. It requires the contributions, matches, and motif clusters, and saves the following in `analysis/motifs/results/`:

- `<path>/{motif,cluster}_{mean,quantile}.tsv`: summaries of each differentiation path, with motifs or clusters as rows and cell states as columns, and heatmaps of the path's cell states.
- `ysl/cluster_mean_trajectories/`: mean contribution of each motif cluster on the differentiation trajectory.
- `cluster_logos/`: consensus PWM of each motif cluster.

Match contributions take about 1 minute to calculate, and are cached in `analysis/motifs/match_contributions.h5`.
