import json as _json
from pathlib import Path as _Path

import pandas as _pandas

# Repository root, so paths do not depend on the working directory
REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

# Reference genome (GRCz11, RefSeq GCF_000002035.6)
###################################################
GENOME_FASTA_PATH = DATA_DIR / 'genome' / 'GCF_000002035.6_GRCz11_genomic.fna.gz'
# Chromosome names to RefSeq accessions used as FASTA record IDs
CHROM_TO_REFSEQ = {f'chr{i}': f'NC_{7111 + i:06d}.7' for i in range(1, 26)}
CHROM_TO_REFSEQ['MT'] = 'NC_002333.2'

# scATAC-seq data
#################
# Pseudobulk ATAC peak accessibility (CPM), peaks x cell type/stage
PEAK_CPM_PATH = DATA_DIR / 'raw' / 'peak.aggregated.counts-CPM.txt'

# Processed peaks, negative regions, and chromosome splits for model training
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TRAINING_DATA_PATH = PROCESSED_DATA_DIR / 'training_data.h5'
CHR_SPLITS_PATH = PROCESSED_DATA_DIR / 'chr_splits.json'

# Length of peaks and negative regions
SEQ_LENGTH = 500

# Cell states
#############
# Cell type, stage, and lineage of each cell state, in the order of model outputs
RESOURCES_DIR = _Path(__file__).resolve().parent / 'resources'
CELL_STATE_METADATA = _pandas.read_csv(RESOURCES_DIR / 'cell_state_metadata.csv', index_col='cell_state')
CELL_STATES = CELL_STATE_METADATA.index.tolist()
STAGES = CELL_STATE_METADATA['stage'].unique().tolist()

# Cell state trajectory: plot coordinates of each cell state, and edges with
# their probabilities
TRAJECTORY_COORDS_PATH = RESOURCES_DIR / 'trajectory_coords.tsv'
TRAJECTORY_EDGES_PATH = RESOURCES_DIR / 'trajectory_edges.tsv'

# Paths through the differentiation trajectory, from high cells to each
# 6-somite cell state, indexed by the cell type of the 6-somite cell state
with open(RESOURCES_DIR / 'differentiation_paths.json') as _f:
    DIFFERENTIATION_PATHS = _json.load(_f)

# Trained models
################
MODELS_DIR = REPO_ROOT / 'models'

# DeepDanio, trained on several chromosome splits
DEEPDANIO_MODEL_DIR = MODELS_DIR / 'deepdanio'
DEEPDANIO_SPLITS = [0, 1, 2]
DEEPDANIO_MODEL_NAME = 'deepdanio_split_{split}'
DEEPDANIO_MODEL_PATHS = {
    split: DEEPDANIO_MODEL_DIR / f'{DEEPDANIO_MODEL_NAME.format(split=split)}.h5' for split in DEEPDANIO_SPLITS
}

# Model predictions
###################
# Ensemble predictions on all peaks, with datasets 'peak_id' and 'pred'
# (n_peaks, n_cell_states)
PREDICTIONS_DIR = DATA_DIR / 'predictions'
ENSEMBLE_PREDICTIONS_PATH = PREDICTIONS_DIR / 'ensemble_predictions.h5'

# Model interpretation
######################
# Contribution scores of cell state-specific peaks in all cell states: actual
# contributions in a single file, and hypothetical contributions in one file
# per cell state, named by cell state index
CONTRIBUTIONS_DIR = DATA_DIR / 'contributions'
# Cell state-specific peaks for which contributions were computed, ranked by
# predicted specificity within each cell state
SELECTED_PEAKS_PATH = CONTRIBUTIONS_DIR / 'selected_peaks.tsv'
CONTRIBUTIONS_PATH = CONTRIBUTIONS_DIR / 'contributions.h5'
HYPOTHETICAL_CONTRIBUTIONS_DIR = CONTRIBUTIONS_DIR / 'hypothetical'
HYPOTHETICAL_CONTRIBUTIONS_NAME = '{cell_state_idx:02d}.h5'

# Motifs discovered with TF-MoDISco in each cell state, in one directory per
# cell state named by cell state index: TF-MoDISco results, trimmed motif CWMs
# and PWMs, seqlets, and motif matches in the cell state's selected peaks
MOTIFS_DIR = DATA_DIR / 'motifs'
MOTIFS_CELL_STATE_DIR_NAME = '{cell_state_idx:02d}'
MODISCO_RESULTS_NAME = 'modisco_results.h5'
CWM_NAME = 'cwm_trimmed.meme'
PWM_NAME = 'pwm_trimmed.meme'
SEQLETS_NAME = 'seqlets.tsv'
MOTIF_MATCHES_NAME = 'motif_matches.tsv'

# Clustering of TF-MoDISco motifs across cell states: cluster of each motif,
# and consensus PWM of each cluster
MOTIF_CLUSTERING_DIR = MOTIFS_DIR / 'clustering'
MOTIF_CLUSTERS_PATH = MOTIF_CLUSTERING_DIR / 'motif_clusters.tsv'
CLUSTER_PWMS_PATH = MOTIF_CLUSTERING_DIR / 'cluster_pwms.meme'

# Manual annotation of the motif clusters
MOTIF_CLUSTER_NAMES_PATH = MOTIFS_DIR / 'motif_cluster_names.tsv'
