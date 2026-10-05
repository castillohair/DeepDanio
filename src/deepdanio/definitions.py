from pathlib import Path as _Path

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

# Trained models
################
MODELS_DIR = REPO_ROOT / 'models'

# DeepDanio, trained on several chromosome splits
DEEPDANIO_MODEL_DIR = MODELS_DIR / 'deepdanio'
DEEPDANIO_SPLITS = [0, 1, 2]
DEEPDANIO_MODEL_PATHS = {
    split: DEEPDANIO_MODEL_DIR / f'deepdanio_split_{split}.h5' for split in DEEPDANIO_SPLITS
}
