"""
Download ensemble predictions of all peaks, selected peaks, their contribution
scores, and motif discovery and clustering results.

Each group of files is downloaded after confirmation, and skipped if already
present. Motif results are distributed as archives of the motifs directory,
which are extracted into the data directory and then deleted.

"""
import tarfile

from deepdanio import definitions, download

# Files: description, and local path, whose name is also the remote name
FILES = [
    ("ensemble predictions of all peaks (180 MB, only needed for peak selection)", definitions.ENSEMBLE_PREDICTIONS_PATH),
    ("selected peaks table (80 MB)", definitions.SELECTED_PEAKS_PATH),
    ("contribution scores (26 GB)", definitions.CONTRIBUTIONS_PATH),
]

# Archives: description, name, and a file whose presence indicates that the archive was extracted
ARCHIVES = [
    (
        "motifs, seqlets, motif clusters, and cluster names (110 MB)",
        'motifs.tar.gz',
        definitions.MOTIF_CLUSTERS_PATH,
    ),
    (
        "TF-MoDISco results (1.1 GB, only needed to rerun motif extraction)",
        'modisco_results.tar.gz',
        definitions.MOTIFS_DIR
        / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=len(definitions.CELL_STATES) - 1)
        / definitions.MODISCO_RESULTS_NAME,
    ),
]

if __name__ == '__main__':
    for description, path in FILES:
        if input(f"Download {description}? (Y/N): ").strip().lower() != 'y':
            continue
        download.download_files([(f'{download.BASE_URL}/{path.name}', path)])

    for description, archive_name, extracted_path in ARCHIVES:
        if input(f"Download {description}? (Y/N): ").strip().lower() != 'y':
            continue
        if extracted_path.exists():
            print(f"{extracted_path} already exists, skipping.")
            continue
        archive_path = definitions.DATA_DIR / archive_name
        download.download_files([(f'{download.BASE_URL}/{archive_name}', archive_path)])
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path) as f:
            f.extractall(definitions.DATA_DIR, filter='data')
        archive_path.unlink()
