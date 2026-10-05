"""
Download data files needed by this repository.

Each group of files is downloaded after confirmation. Files already present
are skipped.

"""
from deepdanio import definitions, download

if __name__ == '__main__':

    # Reference genome (GRCz11)
    ###########################
    if input("Download reference genome (only needed for reprocessing data)? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (
                'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/035/GCF_000002035.6_GRCz11/GCF_000002035.6_GRCz11_genomic.fna.gz',
                definitions.GENOME_FASTA_PATH,
            ),
        ])

    # Raw scATAC-seq data
    #####################
    if input("Download raw scATAC-seq data (only needed for reprocessing data)? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (f'{download.BASE_URL}/peak.aggregated.counts-CPM.txt', definitions.PEAK_CPM_PATH),
        ])

    # Processed training data
    #########################
    # Chromosome splits are tracked in git and not downloaded
    if input("Download processed training data? (Y/N): ").strip().lower() == 'y':
        download.download_files([
            (f'{download.BASE_URL}/training_data.h5', definitions.TRAINING_DATA_PATH),
        ])
