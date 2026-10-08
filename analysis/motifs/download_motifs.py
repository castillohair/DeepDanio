"""
Download motif discovery and clustering results.

Results are distributed as archives of the motifs directory, which are
extracted into the data directory and then deleted. Each archive is downloaded
after confirmation, and skipped if its contents are already present.

"""
import tarfile

from deepdanio import definitions, download

# Archive names, and files whose presence indicates that an archive was extracted
ARCHIVES = [
    (
        "motifs, seqlets, matches, motif clusters, and cluster names (260 MB)",
        'motifs.tar.gz',
        definitions.MOTIF_CLUSTERS_PATH,
    ),
    (
        "TF-MoDISco results (1.1 GB, only needed to rerun motif extraction and scanning)",
        'modisco_results.tar.gz',
        definitions.MOTIFS_DIR
        / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=len(definitions.CELL_STATES) - 1)
        / definitions.MODISCO_RESULTS_NAME,
    ),
]

if __name__ == '__main__':
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
