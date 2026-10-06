import functools

import numpy
import pandas
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle
from numba import njit
from tensorflow import keras

# Reference settings used for all contributions in this repository
REF_SEED = 20231218
N_REFS = 10


# Contribution scores
#####################
# Adapted from
# https://github.com/AvantiShri/colab_notebooks/blob/ecf2909528ff47dbc36c87666b44520f07cbaab2/labmeeting/Oct18/DeepSHAP_Unimodal_Input.ipynb

def dinuc_shuffle_refs(input_modes, seed=REF_SEED, n_refs=N_REFS):
    """
    Generate dinucleotide-shuffled references for one sequence.

    The random generator is reset for every sequence, so each sequence always
    gets the same references.

    Parameters
    ----------
    input_modes : list of numpy.ndarray
        Single-element list with a one-hot sequence of shape (seq_length, 4).
    seed : int, optional
        Random seed.
    n_refs : int, optional
        Number of references.

    Returns
    -------
    list of numpy.ndarray
        Single-element list with references of shape (n_refs, seq_length, 4).

    """
    assert len(input_modes) == 1
    rng = numpy.random.RandomState(seed)
    return [numpy.array([dinuc_shuffle(input_modes[0], rng=rng) for _ in range(n_refs)])]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Compute hypothetical contributions from DeepSHAP multipliers.

    For each possible base at each position, the difference from reference is
    multiplied by the multipliers and summed across bases, giving the
    contribution that base would have if present. Results are averaged across
    references.

    Parameters
    ----------
    mult : list of numpy.ndarray
        Single-element list with multipliers of shape (n_refs, seq_length, 4).
    orig_inp : list of numpy.ndarray
        Single-element list with the one-hot sequence, shape (seq_length, 4).
    bg_data : list of numpy.ndarray
        Single-element list with references of shape (n_refs, seq_length, 4).

    Returns
    -------
    list of numpy.ndarray
        Single-element list with hypothetical contributions of shape
        (seq_length, 4).

    """
    assert len(orig_inp) == 1
    assert len(orig_inp[0].shape) == 2
    projected_hypothetical_contribs = numpy.zeros_like(bg_data[0]).astype('float')
    for base_idx in range(orig_inp[0].shape[-1]):
        hypothetical_input = numpy.zeros_like(orig_inp[0]).astype('float')
        hypothetical_input[:, base_idx] = 1.0
        hypothetical_diff_from_ref = hypothetical_input[None, :, :] - bg_data[0]
        projected_hypothetical_contribs[:, :, base_idx] = numpy.sum(hypothetical_diff_from_ref * mult[0], axis=-1)
    return [numpy.mean(projected_hypothetical_contribs, axis=0)]


def compute_contributions(
        model,
        seqs_onehot,
        output_idx,
        ref_seed=REF_SEED,
        n_refs=N_REFS,
        progress_message=None,
    ):
    """
    Compute hypothetical contribution scores of one model output with DeepSHAP.

    Requires the custom shap version listed in pyproject.toml. Actual
    contributions are obtained by multiplying the result by the one-hot
    sequences.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model to interpret. Ensembles should average outputs with a
        keras.layers.Average layer.
    seqs_onehot : numpy.ndarray
        One-hot sequences with shape (n_seqs, seq_length, 4).
    output_idx : int
        Index of the model output (cell state) to interpret.
    ref_seed : int, optional
        Random seed for dinucleotide-shuffled references.
    n_refs : int, optional
        Number of references per sequence.
    progress_message : int, optional
        If provided, print progress every this many sequences.

    Returns
    -------
    numpy.ndarray
        Hypothetical contributions with shape (n_seqs, seq_length, 4).

    """
    model_output = keras.Model(inputs=model.inputs, outputs=model.output[:, output_idx])

    # Handlers for ops not covered by shap. AddV2 is linear, and dilated
    # convolutions are implemented with SpaceToBatchND / BatchToSpaceND.
    op_handlers = shap.explainers._deep.deep_tf.op_handlers
    op_handlers['AddV2'] = shap.explainers._deep.deep_tf.passthrough
    op_handlers['BatchToSpaceND'] = shap.explainers._deep.deep_tf.linearity_1d(0)
    op_handlers['SpaceToBatchND'] = shap.explainers._deep.deep_tf.linearity_1d(0)

    explainer = shap.DeepExplainer(
        model=model_output,
        data=functools.partial(dinuc_shuffle_refs, seed=ref_seed, n_refs=n_refs),
        combine_mult_and_diffref=combine_mult_and_diffref,
    )
    return explainer.shap_values(seqs_onehot, check_additivity=False, progress_message=progress_message)


# Motif scanning based on contribution weight matrices (CWMs)
#############################################################

@njit
def jaccard_similarity(x, y):
    """
    Continuous Jaccard similarity between two arrays.

    Based on "Technical Note on Transcription Factor Motif Discovery from
    Importance Scores (TF-MoDISco) version 0.5.6.5".

    Parameters
    ----------
    x, y : numpy.ndarray
        Arrays of the same shape.

    Returns
    -------
    float
        Jaccard similarity.

    """
    abs_x = numpy.abs(x)
    abs_y = numpy.abs(y)
    intersection = numpy.sum(numpy.minimum(abs_x, abs_y) * numpy.sign(x) * numpy.sign(y))
    union = numpy.sum(numpy.maximum(abs_x, abs_y))
    return intersection / union


def cwm_similarity(seq_onehot, seq_contrib, cwm, revcomp=True):
    """
    Compare a CWM to the contributions of a sequence at every position.

    Based on "Base-resolution models of transcription-factor binding reveal
    soft motif syntax" (Avsec et al., 2021).

    Parameters
    ----------
    seq_onehot : numpy.ndarray
        One-hot sequence with shape (seq_length, 4).
    seq_contrib : numpy.ndarray
        Hypothetical contributions with shape (seq_length, 4).
    cwm : numpy.ndarray
        Contribution weight matrix with shape (motif_length, 4).
    revcomp : bool, optional
        Whether to also compare the reverse complement of the CWM.

    Returns
    -------
    score_contrib : numpy.ndarray
        Sum of absolute actual contributions in each window, with shape
        (seq_length - motif_length + 1,).
    score_match : numpy.ndarray
        Jaccard similarity between the normalized CWM and the normalized
        actual contributions in each window.
    revcomp_sel : numpy.ndarray
        Whether the reverse complement matched better in each window.

    """
    seq_length = seq_onehot.shape[0]
    motif_length = cwm.shape[0]
    n_windows = seq_length - motif_length + 1

    # Actual contributions
    contrib = seq_contrib * seq_onehot

    score_contrib = numpy.convolve(
        numpy.sum(numpy.abs(contrib), axis=1),
        numpy.ones(motif_length, dtype=int),
        'valid',
    )

    score_match = numpy.zeros(n_windows)
    revcomp_sel = numpy.zeros(n_windows, dtype=bool)
    cwm_norm = cwm / numpy.sum(numpy.abs(cwm))
    cwm_norm_rc = cwm_norm[::-1, ::-1]
    for pos in range(n_windows):
        window = contrib[pos:pos + motif_length, :]
        window_norm = window / numpy.sum(numpy.abs(window))
        score_match_fw = jaccard_similarity(window_norm, cwm_norm)
        if revcomp:
            score_match_rc = jaccard_similarity(window_norm, cwm_norm_rc)
            revcomp_sel[pos] = score_match_rc > score_match_fw
            score_match[pos] = max(score_match_fw, score_match_rc)
        else:
            score_match[pos] = score_match_fw

    return score_contrib, score_match, revcomp_sel


@njit
def scan_pwm(seq_onehot, pwm):
    """
    Sum of PWM probabilities of the sequence bases at every position.

    Parameters
    ----------
    seq_onehot : numpy.ndarray
        One-hot sequence with shape (seq_length, 4).
    pwm : numpy.ndarray
        Position probability matrix with shape (motif_length, 4).

    Returns
    -------
    numpy.ndarray
        Scores with shape (seq_length - motif_length + 1,).

    """
    motif_length = pwm.shape[0]
    n_windows = seq_onehot.shape[0] - motif_length + 1
    score = numpy.zeros(n_windows)
    for pos in range(n_windows):
        score[pos] = numpy.sum(seq_onehot[pos:pos + motif_length, :] * pwm)
    return score


def scan_cwm(
        seq_ids,
        seqs_onehot,
        seqs_contrib,
        cwm,
        pwm,
        match_threshold,
        contrib_threshold,
        pwm_threshold=0.25,
    ):
    """
    Find matches to a motif using its CWM, contributions, and PWM.

    A window is a match if its CWM Jaccard similarity is at least
    `match_threshold`, its total absolute contribution is at least
    `contrib_threshold`, and the mean PWM probability of its bases is at least
    `pwm_threshold`. The default PWM threshold of 0.25 is equivalent to a
    positive log-odds score against a uniform background. The strand of the
    PWM score follows the strand with the best CWM match.

    Parameters
    ----------
    seq_ids : list of str
        Sequence IDs.
    seqs_onehot : numpy.ndarray
        One-hot sequences with shape (n_seqs, seq_length, 4).
    seqs_contrib : array-like
        Hypothetical contributions with shape (n_seqs, seq_length, 4).
    cwm : numpy.ndarray
        Contribution weight matrix with shape (motif_length, 4).
    pwm : numpy.ndarray
        Position probability matrix with shape (motif_length, 4).
    match_threshold : float
        Minimum CWM Jaccard similarity.
    contrib_threshold : float
        Minimum total absolute contribution.
    pwm_threshold : float, optional
        Minimum mean PWM probability.

    Returns
    -------
    pandas.DataFrame
        One row per match, with columns 'seq_id', 'start', 'end', 'revcomp',
        'cwm_contrib', 'cwm_match', and 'pwm_prob'. Positions are 0-based,
        with exclusive ends.

    """
    pwm_rc = pwm[::-1, ::-1]
    motif_length = cwm.shape[0]

    matches = []
    for seq_id, seq_onehot, seq_contrib in zip(seq_ids, seqs_onehot, seqs_contrib):
        score_contrib, score_match, revcomp_sel = cwm_similarity(seq_onehot, seq_contrib, cwm, revcomp=True)
        pwm_score = numpy.where(
            revcomp_sel,
            scan_pwm(seq_onehot, pwm_rc),
            scan_pwm(seq_onehot, pwm),
        ) / motif_length

        is_match = (
            (score_match >= match_threshold)
            & (score_contrib >= contrib_threshold)
            & (pwm_score >= pwm_threshold)
        )
        for start in numpy.flatnonzero(is_match):
            matches.append((
                seq_id,
                start,
                start + motif_length,
                revcomp_sel[start],
                score_contrib[start],
                score_match[start],
                pwm_score[start],
            ))

    return pandas.DataFrame(
        matches,
        columns=['seq_id', 'start', 'end', 'revcomp', 'cwm_contrib', 'cwm_match', 'pwm_prob'],
    )
