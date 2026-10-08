"""
Download ensemble predictions of all peaks, selected peaks, and their
contribution scores.

Each group of files is downloaded after confirmation. Files already present
are skipped.

"""
from deepdanio import definitions, download

if __name__ == '__main__':

    # Ensemble predictions of all peaks
    ###################################
    if input("Download ensemble predictions of all peaks (180 MB, only needed for peak selection)? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (f'{download.BASE_URL}/{definitions.ENSEMBLE_PREDICTIONS_PATH.name}', definitions.ENSEMBLE_PREDICTIONS_PATH),
        ])

    # Selected peaks
    ################
    if input("Download selected peaks table (80 MB)? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (f'{download.BASE_URL}/{definitions.SELECTED_PEAKS_PATH.name}', definitions.SELECTED_PEAKS_PATH),
        ])

    # Actual contributions of selected peaks in all cell states
    ###########################################################
    if input("Download contribution scores (26 GB)? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (f'{download.BASE_URL}/{definitions.CONTRIBUTIONS_PATH.name}', definitions.CONTRIBUTIONS_PATH),
        ])
