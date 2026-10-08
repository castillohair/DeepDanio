"""
Download motif matches in the selected peaks of each cell state.

Matches are distributed as an archive of the motifs directory, which is
extracted into the data directory and then deleted. The download is skipped if
the archive was already extracted.

"""
import tarfile

from deepdanio import definitions, download

ARCHIVE_NAME = 'motif_matches.tar.gz'
# File whose presence indicates that the archive was extracted
EXTRACTED_PATH = (
    definitions.MOTIFS_DIR
    / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=len(definitions.CELL_STATES) - 1)
    / definitions.MOTIF_MATCHES_NAME
)

if __name__ == '__main__':
    if EXTRACTED_PATH.exists():
        print(f"{EXTRACTED_PATH} already exists, skipping.")
    else:
        archive_path = definitions.DATA_DIR / ARCHIVE_NAME
        download.download_files([(f'{download.BASE_URL}/{ARCHIVE_NAME}', archive_path)])
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path) as f:
            f.extractall(definitions.DATA_DIR, filter='data')
        archive_path.unlink()
