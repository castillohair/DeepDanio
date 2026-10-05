import gzip

import Bio.SeqIO

from deepdanio import definitions


def load_genome(fasta_path=definitions.GENOME_FASTA_PATH):
    """
    Load a gzipped genome FASTA file into memory.

    Parameters
    ----------
    fasta_path : str or pathlib.Path, optional
        Path to the gzipped FASTA file.

    Returns
    -------
    dict
        Sequences as strings, keyed by FASTA record ID.

    """
    with gzip.open(fasta_path, 'rt') as handle:
        genome = {record.id: str(record.seq) for record in Bio.SeqIO.parse(handle, 'fasta')}
    return genome


def get_sequence(genome, chrom, start, end):
    """
    Extract an uppercase genome sequence.

    Parameters
    ----------
    genome : dict
        Genome sequences keyed by RefSeq accession.
    chrom : str
        Chromosome name (e.g. 'chr1').
    start, end : int
        Zero-based, half-open coordinates.

    Returns
    -------
    str
        Uppercase sequence.

    """
    chrom_seq = genome[definitions.CHROM_TO_REFSEQ[chrom]]
    if end >= len(chrom_seq):
        raise ValueError(f"End position {end} out of range for {chrom} (length {len(chrom_seq)}).")
    return chrom_seq[start:end].upper()
