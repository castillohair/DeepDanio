import gzip

import Bio.SeqIO

from deepdanio import definitions


def load_genome(chroms=None, fasta_path=definitions.GENOME_FASTA_PATH):
    """
    Load chromosome sequences from a gzipped genome FASTA file.

    Parameters
    ----------
    chroms : list of str, optional
        Chromosome names to load (e.g. 'chr1'). Loading only the necessary
        chromosomes saves time and memory. Default is all chromosomes in
        CHROM_TO_REFSEQ.
    fasta_path : str or pathlib.Path, optional
        Path to the gzipped FASTA file.

    Returns
    -------
    dict
        Uppercase sequences keyed by chromosome name.

    """
    if chroms is None:
        chroms = list(definitions.CHROM_TO_REFSEQ)
    unknown = [chrom for chrom in chroms if chrom not in definitions.CHROM_TO_REFSEQ]
    if unknown:
        raise ValueError(f"Unknown chromosomes: {unknown}")

    # FASTA records are identified by RefSeq accession
    refseq_to_chrom = {definitions.CHROM_TO_REFSEQ[chrom]: chrom for chrom in chroms}
    genome = {}
    with gzip.open(fasta_path, 'rt') as handle:
        for record in Bio.SeqIO.parse(handle, 'fasta'):
            if record.id in refseq_to_chrom:
                genome[refseq_to_chrom[record.id]] = str(record.seq).upper()
                if len(genome) == len(refseq_to_chrom):
                    break

    missing = [chrom for chrom in chroms if chrom not in genome]
    if missing:
        raise ValueError(f"Chromosomes not found in {fasta_path}: {missing}")
    return genome


def get_sequence(genome, chrom, start, end):
    """
    Extract a genome sequence.

    Parameters
    ----------
    genome : dict
        Sequences keyed by chromosome name.
    chrom : str
        Chromosome name (e.g. 'chr1').
    start, end : int
        Zero-based, half-open coordinates.

    Returns
    -------
    str
        Sequence.

    """
    if chrom not in genome:
        raise ValueError(f"{chrom} not loaded. Loaded chromosomes: {list(genome)}")
    chrom_length = len(genome[chrom])
    if not 0 <= start < end <= chrom_length:
        raise ValueError(f"[{start:,}, {end:,}) is outside {chrom} (length {chrom_length:,}).")
    return genome[chrom][start:end]
