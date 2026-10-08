# Motif contributions

This folder finds matches to the discovered motifs in the selected peaks of each cell state, and quantifies how much they contribute to predicted accessibility in all cell states, summarized along each of the 29 differentiation paths.

It requires the selected peaks, contributions, motifs, seqlets, and motif clusters from [`analysis/motif_discovery`](../motif_discovery/), either recomputed or downloaded with `download_results.py` in that folder.

## Downloading results

`download_matches.py` downloads the motif matches of each cell state (150 MB) into `data/motifs/{cell_state_idx}/motif_matches.tsv`, with the same columns as the seqlets table: `motif`, `peak_id`, `start`, `end` (0-based, exclusive), `revcomp`, `cwm_contrib`, `cwm_match`, and `pwm_prob`.

## Reproducing results

Scripts are run in the following order, from the repository root.

### Motif scanning

```
usage: scan_motifs.py [-h] --cell-state CELL_STATE [--motifs-dir MOTIFS_DIR]
```

Scans the selected peaks of one cell state, given by name or index, for matches to each of the cell state's motifs, using the actual contributions in that cell state. For example, for adaxial cells (6-somite):

```shell
python analysis/motif_contributions/scan_motifs.py --cell-state 91
```

A match requires a CWM similarity of at least the 0.2 quantile of the motif's seqlets, a total absolute contribution of at least the minimum of the motif's seqlets, and a mean PWM probability of at least 0.25. This requires the processed training data (see `data/download_data.py`).

Scanning runs on a single CPU core and takes about 15 to 30 minutes per cell state on an Intel Xeon Platinum CPU, with about 1.3 GB of memory. Cell states are independent, so we recommend processing them in parallel.

Rerunning the script reproduces the released results.

### Motif contributions

[`motif_contributions.ipynb`](./motif_contributions.ipynb) calculates the contribution of each match to the top 5 positive motifs of each cell state in all cell states, and summarizes them per motif and per motif cluster along each differentiation path, as means and 90th percentiles. It saves the following in `analysis/motif_contributions/results/`:

- `<path>/{motif,cluster}_{mean,quantile}.tsv`: summaries of each differentiation path, with motifs or clusters as rows and cell states as columns, and heatmaps of the path's cell states.
- `ysl/cluster_mean_trajectories/`: mean contribution of each motif cluster on the differentiation trajectory.
- `cluster_logos/`: consensus PWM of each motif cluster.

Match contributions take about 1 minute to calculate, and are cached in `analysis/motif_contributions/match_contributions.h5`.
