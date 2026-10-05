from pathlib import Path as _Path

# Repository root, so paths do not depend on the working directory
REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

# Pseudobulk ATAC peak accessibility (CPM), peaks x cell type/stage
PEAK_CPM_PATH = DATA_DIR / 'peak.aggregated.counts-CPM.txt'
