"""
Scan the selected peaks of one cell state for matches to the cell state's motifs.

Peaks are scanned for matches to each trimmed motif using the actual
contributions in the cell state. A match requires a CWM similarity of at least
the MATCH_QUANTILE quantile of the motif's seqlets, a total absolute
contribution of at least the minimum of the motif's seqlets, and a mean PWM
probability of at least 0.25.

Inputs are the selected peaks, the training data sequences, the contributions,
and the cell state's trimmed CWMs and PWMs and seqlets, read from
'{motifs_dir}/{cell_state_idx:02d}/'.

Matches are saved to 'motif_matches.tsv' in the same directory, with columns
'motif', 'peak_id', 'start', 'end' (0-based, exclusive), 'revcomp',
'cwm_contrib', 'cwm_match', and 'pwm_prob'.

"""
import argparse
from pathlib import Path

import pandas

from deepdanio import data, definitions, interpret, motif, sequence

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
    motifs_dir = args.motifs_dir / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=cell_state_idx)

    print(f"Cell state {cell_state_idx}: {cell_state}")

    # Peaks selected in the cell state, by decreasing specificity
    selected_df = pandas.read_csv(definitions.SELECTED_PEAKS_PATH, sep='\t')
    peak_ids = selected_df.loc[selected_df['cell_state'] == cell_state].sort_values('rank')['peak_id'].tolist()
    peaks_df, _ = data.load_training_data('peaks')
    seqs_onehot = sequence.one_hot_encode(peaks_df.loc[peak_ids, 'sequence'].values)
    # Actual contributions with shape (n_peaks, seq_length, 1)
    contribs = data.load_contributions(peak_ids, [cell_state])[:, 0, :, None]

    # Motifs and seqlets
    cwms = motif.load_meme(motifs_dir / definitions.CWM_NAME)
    pwms = motif.load_meme(motifs_dir / definitions.PWM_NAME)
    seqlets_df = pandas.read_csv(motifs_dir / definitions.SEQLETS_NAME, sep='\t')

    print(f"Scanning {len(cwms)} motifs in {len(peak_ids):,} peaks...")
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
    match_df.to_csv(motifs_dir / definitions.MOTIF_MATCHES_NAME, sep='\t', index=False)
    print(f"Found {len(match_df):,} matches. Results saved to {motifs_dir}.")
