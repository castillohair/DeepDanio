import numpy

# One-hot encoding of each byte value. Bases other than ACGT are encoded as zeros.
_ONEHOT_LOOKUP = numpy.zeros((256, 4), dtype=numpy.float32)
for _idx, _base in enumerate('ACGT'):
    _ONEHOT_LOOKUP[ord(_base), _idx] = 1
    _ONEHOT_LOOKUP[ord(_base.lower()), _idx] = 1

_REVCOMP_TABLE = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def one_hot_encode(seqs):
    """
    One-hot encode DNA sequences of equal length.

    Parameters
    ----------
    seqs : array-like of str
        Sequences, all of the same length.

    Returns
    -------
    numpy.ndarray
        Array with shape (n_seqs, seq_length, 4), with channels ordered A, C,
        G, T.

    """
    seqs_bytes = numpy.asarray(seqs, dtype='S')
    seq_length = seqs_bytes.dtype.itemsize
    if not all(len(s) == seq_length for s in seqs_bytes):
        raise ValueError("All sequences must have the same length.")
    seqs_uint8 = seqs_bytes.view(numpy.uint8).reshape(len(seqs_bytes), seq_length)
    return _ONEHOT_LOOKUP[seqs_uint8]


def revcomp(seq):
    """
    Reverse complement a DNA sequence.

    Parameters
    ----------
    seq : str
        Sequence.

    Returns
    -------
    str
        Reverse complement.

    """
    return seq.translate(_REVCOMP_TABLE)[::-1]
