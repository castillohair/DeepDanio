"""
Generate model training data from pseudobulk scATAC-seq peaks.

Outputs:
- Processed HDF5 file with peaks (coordinates, sequences, normalized signal)
  and random genomic negative regions (coordinates, sequences).
- JSON file with chromosome-based train/validation/test splits.

"""
import json

import h5py
import numpy
import pandas
import prtpy

from deepdanio import data, definitions, genome

# Negative regions: number, and minimum distance to peaks and each other
N_NEGATIVES = 200000
NEGATIVE_SPACING = 100
NEGATIVES_RANDOM_SEED = 0

# Number of chromosome groups used for data splits
N_CHR_GROUPS = 10

if __name__ == '__main__':

    # Load and normalize signal
    ###########################
    print("Loading CPM data...")
    cpm_df = pandas.read_csv(definitions.PEAK_CPM_PATH, sep='\t')
    cell_states = cpm_df.columns.tolist()
    print(f"{len(cpm_df):,} peaks, {len(cell_states)} cell states.")
    assert numpy.allclose(cpm_df.sum(axis=0), 1e6), "CPM columns do not sum to 1e6."
    assert cell_states == definitions.CELL_STATES, "CPM columns do not match the cell state metadata."

    # Log transform, using the minimum nonzero value of each cell state as pseudocount
    cpm = cpm_df.values
    pseudocounts = numpy.array([cpm[cpm[:, i] > 0, i].min() for i in range(cpm.shape[1])])
    log_cpm = numpy.log10(cpm + pseudocounts)

    print("Quantile normalizing...")
    signal = data.quantile_normalize(log_cpm)

    # Peak coordinates and sequences
    ################################
    # Peak IDs are "<chromosome number>-<start>-<end>"
    peaks_df = cpm_df.index.to_series().str.split('-', expand=True)
    peaks_df.columns = ['chr', 'start', 'end']
    peaks_df['chr'] = 'chr' + peaks_df['chr']
    peaks_df['start'] = peaks_df['start'].astype(int)
    peaks_df['end'] = peaks_df['end'].astype(int)
    assert ((peaks_df['end'] - peaks_df['start']) == definitions.SEQ_LENGTH).all()

    print("Loading genome...")
    genome_seqs = genome.load_genome()

    print("Extracting peak sequences...")
    peaks_df['sequence'] = [
        genome.get_sequence(genome_seqs, chrom, start, end)
        for chrom, start, end in zip(peaks_df['chr'], peaks_df['start'], peaks_df['end'])
    ]
    assert (peaks_df['sequence'].str.len() == definitions.SEQ_LENGTH).all()

    # Peaks should be sorted and non-overlapping within each chromosome
    chroms = peaks_df['chr'].unique()
    for chrom in chroms:
        chrom_df = peaks_df[peaks_df['chr'] == chrom]
        assert (chrom_df['start'].diff()[1:] >= 0).all(), f"Peaks not sorted in {chrom}."
        assert (chrom_df['start'].values[1:] > chrom_df['end'].values[:-1]).all(), f"Overlapping peaks in {chrom}."

    # Negative regions
    ##################
    # Sampled uniformly from the genome space not within NEGATIVE_SPACING of a
    # peak, with counts per chromosome proportional to chromosome length.
    seq_len = definitions.SEQ_LENGTH
    chrom_lengths = numpy.array([len(genome_seqs[c]) for c in chroms])
    n_negatives_per_chrom = numpy.round(N_NEGATIVES * chrom_lengths / chrom_lengths.sum()).astype(int)

    numpy.random.seed(NEGATIVES_RANDOM_SEED)
    negatives = []
    for chrom, chrom_length, n_negatives_chrom in zip(chroms, chrom_lengths, n_negatives_per_chrom):
        print(f"Sampling {n_negatives_chrom:,} negative regions from {chrom}...")
        chrom_df = peaks_df[peaks_df['chr'] == chrom]

        # Start positions available for sampling
        is_available = numpy.ones(chrom_length - seq_len, dtype=bool)
        for start, end in zip(chrom_df['start'], chrom_df['end']):
            is_available[max(start - seq_len - NEGATIVE_SPACING, 0):end + NEGATIVE_SPACING] = False
        available_starts = numpy.flatnonzero(is_available)

        # Sample indices into available_starts, resampling those too close to each other
        sampled_idx = numpy.array([], dtype=int)
        while len(sampled_idx) < n_negatives_chrom:
            new_idx = numpy.random.choice(
                len(available_starts),
                size=n_negatives_chrom - len(sampled_idx),
                replace=False,
            )
            sampled_idx = numpy.sort(numpy.concatenate((sampled_idx, new_idx)))
            too_close = numpy.diff(sampled_idx) <= seq_len + NEGATIVE_SPACING
            sampled_idx = numpy.concatenate((sampled_idx[:1], sampled_idx[1:][~too_close]))

        for start in available_starts[sampled_idx]:
            negatives.append({
                'id': f"{chrom[3:]}-{start}-{start + seq_len}",
                'chr': chrom,
                'start': start,
                'end': start + seq_len,
                'sequence': genome.get_sequence(genome_seqs, chrom, start, start + seq_len),
            })
    negatives_df = pandas.DataFrame(negatives).set_index('id')
    assert (negatives_df['sequence'].str.len() == seq_len).all()

    # Peaks and negatives should not overlap
    regions_df = pandas.concat([peaks_df, negatives_df])
    for chrom in chroms:
        chrom_df = regions_df[regions_df['chr'] == chrom].sort_values('start')
        assert (chrom_df['start'].values[1:] > chrom_df['end'].values[:-1]).all(), f"Overlapping regions in {chrom}."

    # Chromosome splits
    ###################
    # Chromosomes are grouped so that groups have similar numbers of peaks.
    # In split i, group i is used for testing and group i + 1 for validation.
    n_peaks_per_chrom = peaks_df.groupby('chr').size().to_dict()
    chr_groups = prtpy.partition(
        algorithm=prtpy.partitioning.greedy,
        numbins=N_CHR_GROUPS,
        items=n_peaks_per_chrom,
    )
    chr_splits = []
    for group_idx in range(N_CHR_GROUPS):
        test_chrs = chr_groups[group_idx]
        val_chrs = chr_groups[(group_idx + 1) % N_CHR_GROUPS]
        train_chrs = []
        for chr_group in chr_groups:
            if chr_group != test_chrs and chr_group != val_chrs:
                train_chrs.extend(chr_group)
        chr_splits.append({'train': train_chrs, 'val': val_chrs, 'test': test_chrs})

    for split_idx, chr_split in enumerate(chr_splits):
        n_train, n_val, n_test = [
            peaks_df['chr'].isin(chr_split[subset]).sum() for subset in ['train', 'val', 'test']
        ]
        print(f"Split {split_idx}: {n_train:,} / {n_val:,} / {n_test:,} train/val/test peaks.")

    # Save
    ######
    definitions.PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    print(f"Saving {definitions.TRAINING_DATA_PATH}...")
    with h5py.File(definitions.TRAINING_DATA_PATH, 'w') as f:
        f.create_dataset('cell_states', data=numpy.array(cell_states, dtype='S'))
        for group_name, regions_df in [('peaks', peaks_df), ('negatives', negatives_df)]:
            h5_group = f.create_group(group_name)
            h5_group.create_dataset('id', data=regions_df.index.values.astype('S'))
            h5_group.create_dataset('chr', data=regions_df['chr'].values.astype('S'))
            h5_group.create_dataset('start', data=regions_df['start'].values.astype('int64'))
            h5_group.create_dataset('end', data=regions_df['end'].values.astype('int64'))
            h5_group.create_dataset(
                'sequence',
                data=regions_df['sequence'].values.astype(f'S{seq_len}'),
                compression='gzip',
            )
        f['peaks'].create_dataset('signal', data=signal.astype('float32'), compression='gzip')

    print(f"Saving {definitions.CHR_SPLITS_PATH}...")
    # One line per chromosome list, for readability
    split_strs = []
    for chr_split in chr_splits:
        subset_strs = [f'        "{subset}": {json.dumps(chrs)}' for subset, chrs in chr_split.items()]
        split_strs.append('    {\n' + ',\n'.join(subset_strs) + '\n    }')
    with open(definitions.CHR_SPLITS_PATH, 'w') as f:
        f.write('[\n' + ',\n'.join(split_strs) + '\n]\n')

    print("Done.")
