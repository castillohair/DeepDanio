"""
Download data files needed by this repository.

Each group of files is downloaded after confirmation. Files already present
are skipped.

"""
import shutil
import urllib.request

from deepdanio import definitions

# TODO: replace with the actual repository (e.g. Zenodo record) once files are uploaded
BASE_URL = 'https://example.org/deepdanio'


def ask_and_download(message, url_path_list):
    reply = input(f"{message} (Y/N): ").strip().lower()
    if reply != 'y':
        return

    for url, dest_path in url_path_list:
        if dest_path.exists():
            print(f"{dest_path} already exists, skipping.")
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        # Download to a temporary file so interrupted downloads are not mistaken for complete ones
        tmp_path = dest_path.with_name(dest_path.name + '.part')
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp_path, 'wb') as f:
            shutil.copyfileobj(response, f)
        tmp_path.rename(dest_path)

    print("Done.")
    print()


if __name__ == '__main__':

    # Reference genome (GRCz11)
    ###########################
    url_path_list = [
        (
            'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/035/GCF_000002035.6_GRCz11/GCF_000002035.6_GRCz11_genomic.fna.gz',
            definitions.GENOME_FASTA_PATH,
        ),
    ]
    ask_and_download("Download reference genome (only needed for reprocessing data)?", url_path_list)

    # Raw scATAC-seq data
    #####################
    url_path_list = [
        (f'{BASE_URL}/peak.aggregated.counts-CPM.txt', definitions.PEAK_CPM_PATH),
    ]
    ask_and_download("Download raw scATAC-seq data (only needed for reprocessing data)?", url_path_list)

    # Processed training data
    #########################
    # Chromosome splits are tracked in git and not downloaded
    url_path_list = [
        (f'{BASE_URL}/training_data.h5', definitions.TRAINING_DATA_PATH),
    ]
    ask_and_download("Download processed training data?", url_path_list)
