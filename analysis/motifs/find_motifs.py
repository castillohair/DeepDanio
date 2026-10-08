"""
Discover motifs in the selected peaks of one cell state, and find their matches.

1. Motif discovery: TF-MoDISco is run on the hypothetical contributions of the
   peaks selected in the cell state, in order of decreasing specificity, within
   the central MODISCO_WINDOW positions. This step is skipped if TF-MoDISco
   results are already present. Reproducing the released results requires
   leidenalg < 0.12.
2. Motif extraction: the CWM and PWM of each pattern are trimmed to positions
   with a total absolute contribution of at least TRIM_THRESHOLD times the
   maximum, plus TRIM_FLANK positions on each side. Seqlets are mapped to
   positions in their peaks, and scored against their trimmed motif as in the
   following step.
3. Motif scanning: the selected peaks are scanned for matches to each trimmed
   motif using actual contributions. A match requires a CWM similarity of at
   least the MATCH_QUANTILE quantile of the motif's seqlets, a total absolute
   contribution of at least the minimum of the motif's seqlets, and a mean PWM
   probability of at least 0.25.

Inputs are the selected peaks, the training data sequences, and the
contributions. The cell state's hypothetical contributions are only needed to
run TF-MoDISco.

Outputs are saved in '{motifs_dir}/{cell_state_idx:02d}/': TF-MoDISco results
('modisco_results.h5'), trimmed CWMs and PWMs ('cwm_trimmed.meme',
'pwm_trimmed.meme') with motif IDs '{pos/neg}_patterns_pattern_{idx}', and
tables of seqlets and matches ('seqlets.tsv', 'motif_matches.tsv') with columns
'motif', 'peak_id', 'start', 'end' (0-based, exclusive), 'revcomp',
'cwm_contrib', 'cwm_match', and 'pwm_prob'.

"""
import argparse
import re
from pathlib import Path

import h5py
import modiscolite
import numpy
import pandas

from deepdanio import data, definitions, interpret, motif, sequence

# TF-MoDISco settings: maximum seqlets per metacluster, window around the
# sequence center where seqlets are searched, and other settings as in the
# TF-MoDISco command line tool
MAX_SEQLETS = 25000
MODISCO_WINDOW = 400
MODISCO_KWARGS = dict(sliding_window_size=20, flank_size=5, target_seqlet_fdr=0.05, n_leiden_runs=2)

# Motif trimming
TRIM_THRESHOLD = 0.3
TRIM_FLANK = 4

# Quantile of seqlet CWM match scores used as match threshold
MATCH_QUANTILE = 0.2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-state', type=str, required=True, help='Cell state name or index.')
    parser.add_argument(
        '--motifs-dir',
        type=Path,
        default=definitions.MOTIFS_DIR,
        help='Directory with one motif directory per cell state.',
    )
    args = parser.parse_args()

    cell_state_idx = int(args.cell_state) if args.cell_state.isdigit() else definitions.CELL_STATES.index(args.cell_state)
    cell_state = definitions.CELL_STATES[cell_state_idx]
    output_dir = args.motifs_dir / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=cell_state_idx)
    output_dir.mkdir(parents=True, exist_ok=True)
    modisco_path = output_dir / definitions.MODISCO_RESULTS_NAME

    print(f"Cell state {cell_state_idx}: {cell_state}")
    print(f"Output: {output_dir}")

    # Peaks selected in the cell state, by decreasing specificity
    selected_df = pandas.read_csv(definitions.SELECTED_PEAKS_PATH, sep='\t')
    peak_ids = selected_df.loc[selected_df['cell_state'] == cell_state].sort_values('rank')['peak_id'].tolist()
    peaks_df, _ = data.load_training_data('peaks')
    seqs = peaks_df.loc[peak_ids, 'sequence'].values
    seqs_onehot = sequence.one_hot_encode(seqs)
    # Positions of the TF-MoDISco window, to which seqlet coordinates are relative
    window_offset = (definitions.SEQ_LENGTH - MODISCO_WINDOW) // 2
    window = slice(window_offset, window_offset + MODISCO_WINDOW)

    # Motif discovery
    #################
    if modisco_path.exists():
        print(f"Using TF-MoDISco results in {modisco_path}.")
    else:
        print(f"Running TF-MoDISco on {len(peak_ids):,} peaks of {cell_state}...")
        hyp_contribs, _ = data.load_hypothetical_contributions(cell_state, peak_ids)
        # TF-MoDISco results depend on the memory layout of its inputs. Arrays
        # are stored as (n, 4, L) and passed as (n, L, 4) views, as in the
        # TF-MoDISco command line tool, which produced the released results.
        hyp_contribs = numpy.ascontiguousarray(hyp_contribs[:, window].transpose(0, 2, 1), dtype='float32')
        modisco_onehot = numpy.ascontiguousarray(seqs_onehot[:, window].transpose(0, 2, 1), dtype='float32')
        pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
            hypothetical_contribs=hyp_contribs.transpose(0, 2, 1),
            one_hot=modisco_onehot.transpose(0, 2, 1),
            max_seqlets_per_metacluster=MAX_SEQLETS,
            **MODISCO_KWARGS,
        )
        modiscolite.io.save_hdf5(str(modisco_path), pos_patterns, neg_patterns, MODISCO_WINDOW)

    # Motif extraction
    ##################
    print("Extracting motifs and seqlets...")
    cwms = {}
    pwms = {}
    nsites = {}
    seqlet_rows = []
    with h5py.File(modisco_path, 'r') as f:
        for metacluster_name in ['pos_patterns', 'neg_patterns']:
            if metacluster_name not in f:
                continue
            pattern_names = sorted(f[metacluster_name], key=lambda name: int(re.search(r'\d+$', name).group()))
            for pattern_name in pattern_names:
                pattern = f[metacluster_name][pattern_name]
                motif_id = f'{metacluster_name}_{pattern_name}'

                # Trim positions with low contribution
                cwm = pattern['contrib_scores'][:]
                pos_contrib = numpy.abs(cwm).sum(axis=1)
                pass_idx = numpy.flatnonzero(pos_contrib >= pos_contrib.max() * TRIM_THRESHOLD)
                trim_start = max(pass_idx.min() - TRIM_FLANK, 0)
                trim_end = min(pass_idx.max() + TRIM_FLANK + 1, len(cwm))
                cwms[motif_id] = cwm[trim_start:trim_end]
                pwms[motif_id] = pattern['sequence'][trim_start:trim_end]

                # Seqlets, with sequences and contributions already reverse
                # complemented if needed
                seqlets = {key: pattern['seqlets'][key][:] for key in pattern['seqlets']}
                nsites[motif_id] = int(seqlets['n_seqlets'][0])
                for seqlet_idx in range(nsites[motif_id]):
                    peak_idx = seqlets['example_idx'][seqlet_idx]
                    seqlet_start = seqlets['start'][seqlet_idx] + window_offset
                    seqlet_end = seqlets['end'][seqlet_idx] + window_offset
                    is_revcomp = bool(seqlets['is_revcomp'][seqlet_idx])

                    # Trimmed seqlet coordinates in the peak
                    if is_revcomp:
                        start, end = seqlet_end - trim_end, seqlet_end - trim_start
                    else:
                        start, end = seqlet_start + trim_start, seqlet_start + trim_end
                    seqlet_onehot = seqlets['sequence'][seqlet_idx][trim_start:trim_end]
                    seqlet_contrib = seqlets['contrib_scores'][seqlet_idx][trim_start:trim_end]
                    peak_seq = seqs[peak_idx][start:end]
                    if is_revcomp:
                        peak_seq = sequence.revcomp(peak_seq)
                    if not numpy.array_equal(sequence.one_hot_encode([peak_seq])[0], seqlet_onehot):
                        raise ValueError(f"Seqlet {seqlet_idx} of {motif_id} does not match its peak sequence.")

                    score_contrib, score_match, _ = interpret.cwm_similarity(
                        seqlet_onehot, seqlet_contrib, cwms[motif_id], revcomp=False,
                    )
                    seqlet_rows.append((
                        motif_id,
                        peak_ids[peak_idx],
                        start,
                        end,
                        is_revcomp,
                        score_contrib[0],
                        score_match[0],
                        numpy.sum(seqlet_onehot * pwms[motif_id]) / len(pwms[motif_id]),
                    ))

    table_columns = ['motif', 'peak_id', 'start', 'end', 'revcomp', 'cwm_contrib', 'cwm_match', 'pwm_prob']
    seqlets_df = pandas.DataFrame(seqlet_rows, columns=table_columns)
    motif.save_meme(cwms, output_dir / definitions.CWM_NAME, nsites=nsites)
    motif.save_meme(pwms, output_dir / definitions.PWM_NAME, nsites=nsites)
    seqlets_df.to_csv(output_dir / definitions.SEQLETS_NAME, sep='\t', index=False)
    print(f"Extracted {len(cwms)} motifs and {len(seqlets_df):,} seqlets.")

    # Motif scanning
    ################
    print("Scanning motifs...")
    # Motifs as saved, so that scanning with the saved files gives the same results
    cwms = motif.load_meme(output_dir / definitions.CWM_NAME)
    pwms = motif.load_meme(output_dir / definitions.PWM_NAME)
    # Actual contributions with shape (n_peaks, seq_length, 1)
    contribs = data.load_contributions(peak_ids, [cell_state])[:, 0, :, None]
    match_dfs = []
    for motif_id in cwms:
        motif_seqlets_df = seqlets_df[seqlets_df['motif'] == motif_id]
        match_df = interpret.scan_cwm(
            peak_ids,
            seqs_onehot,
            contribs,
            cwms[motif_id],
            pwms[motif_id],
            match_threshold=motif_seqlets_df['cwm_match'].quantile(MATCH_QUANTILE),
            contrib_threshold=motif_seqlets_df['cwm_contrib'].min(),
        )
        match_df.insert(0, 'motif', motif_id)
        match_dfs.append(match_df)
    match_df = pandas.concat(match_dfs, ignore_index=True).rename(columns={'seq_id': 'peak_id'})
    match_df.to_csv(output_dir / definitions.MOTIF_MATCHES_NAME, sep='\t', index=False)
    print(f"Found {len(match_df):,} matches. Results saved to {output_dir}.")
