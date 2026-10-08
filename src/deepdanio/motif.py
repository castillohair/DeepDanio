from memelite.io import read_meme

MEME_HEADER = """MEME version 5

ALPHABET= ACGT

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""


def load_meme(filepath):
    """
    Load motifs from a MEME file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the MEME file.

    Returns
    -------
    dict
        Matrices with shape (motif_length, 4), indexed by motif ID. Motif
        names following the ID in the MEME file are ignored.

    """
    return {key.split()[0]: matrix.T for key, matrix in read_meme(str(filepath)).items()}


def save_meme(motifs, filepath, nsites=None):
    """
    Save motifs to a MEME file with uniform background frequencies.

    Motif IDs are also written as alternative names, which some tools require.

    Parameters
    ----------
    motifs : dict
        Matrices with shape (motif_length, 4), indexed by motif ID.
    filepath : str or pathlib.Path
        Path to the MEME file.
    nsites : dict, optional
        Number of sites of each motif, indexed by motif ID. Default is 1.

    """
    meme_str = MEME_HEADER
    for motif_id, matrix in motifs.items():
        motif_nsites = 1 if nsites is None else nsites[motif_id]
        meme_str += f"MOTIF {motif_id} {motif_id}\n"
        meme_str += f"letter-probability matrix: alength= 4 w= {len(matrix)} nsites= {motif_nsites}\n"
        for row in matrix:
            meme_str += ' '.join(f'{v:.6f}' for v in row) + '\n'
        meme_str += '\n'

    with open(filepath, 'w') as f:
        f.write(meme_str)
