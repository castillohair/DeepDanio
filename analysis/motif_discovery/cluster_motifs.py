"""
Cluster TF-MoDISco motifs of all cell states by PWM similarity.

The N_MOTIFS_PER_CELL_STATE motifs with the most seqlets in each cell state,
with ties broken by placing positive motifs first and then by alphabetical
order of motif ID, are clustered with matrix-clustering from RSAT (stand-alone version,
https://github.com/jaimicore/matrix-clustering_stand-alone), using complete
linkage and a normalized correlation threshold of NCOR_THRESHOLD. Motif IDs in
the clustering have the format
'{cell_state_idx}_{cell_state}_{pos/neg}_{pattern_idx}', with spaces in cell
state names replaced by underscores and '/' replaced by '_or_'.

Outputs, in the motif clustering directory:
- Input for matrix-clustering: a MEME file with the motifs to cluster, and a
  table listing it.
- matrix-clustering results, with prefix 'matrix_clustering'.
- A table with the cluster of each motif, with columns 'cell_state', 'motif',
  and 'cluster'.
- A MEME file with the consensus PWM of each cluster.

If matrix-clustering results are not present and the path to
matrix-clustering.R is not given, the input files are written and the command
to run matrix-clustering from the motif clustering directory is printed.

The original clustering is reproduced exactly by matrix-clustering commit
a749c3f with R 4.3.3. Commits before 2024-02-22 give different clusters with
complete linkage. With recent versions of the R package parallelly,
matrix-clustering fails with more than one thread.

"""
import argparse
import re
import subprocess
from pathlib import Path

import numpy
import pandas
from Bio import motifs as bio_motifs

from deepdanio import definitions, motif

N_MOTIFS_PER_CELL_STATE = 10
LINKAGE = 'complete'
NCOR_THRESHOLD = 0.55

# Collection name of the motifs in matrix-clustering, used as prefix of motif IDs in its results
COLLECTION_NAME = 'abcd'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--motifs-dir',
        type=Path,
        default=definitions.MOTIFS_DIR,
        help='Directory with one motif directory per cell state.',
    )
    parser.add_argument(
        '--clustering-dir',
        type=Path,
        default=definitions.MOTIF_CLUSTERING_DIR,
        help='Output directory.',
    )
    parser.add_argument('--matrix-clustering', type=Path, help='Path to matrix-clustering.R.')
    parser.add_argument('--n-threads', type=int, default=1, help='Threads used by matrix-clustering.')
    args = parser.parse_args()

    args.clustering_dir.mkdir(parents=True, exist_ok=True)
    input_meme_path = args.clustering_dir / 'motifs.meme'
    input_table_path = args.clustering_dir / 'motifs_table.txt'
    results_prefix = args.clustering_dir / 'matrix_clustering'
    clusters_tab_path = Path(f'{results_prefix}_tables') / 'clusters.tab'
    root_motifs_path = Path(f'{results_prefix}_motifs') / 'root_motifs' / 'Root_motifs.tf'

    # Motifs with the most seqlets in each cell state
    pwms = {}
    nsites = {}
    for cell_state_idx, cell_state in enumerate(definitions.CELL_STATES):
        cell_state_dir = args.motifs_dir / definitions.MOTIFS_CELL_STATE_DIR_NAME.format(cell_state_idx=cell_state_idx)
        cell_state_pwms = motif.load_meme(cell_state_dir / definitions.PWM_NAME)
        seqlet_counts = pandas.read_csv(cell_state_dir / definitions.SEQLETS_NAME, sep='\t')['motif'].value_counts()
        # Ties are broken by placing positive motifs first, then by alphabetical order of motif ID
        top_motif_ids = sorted(
            cell_state_pwms,
            key=lambda motif_id: (-seqlet_counts[motif_id], not motif_id.startswith('pos'), motif_id),
        )
        cell_state_name = cell_state.replace('/', ' or ').replace(' ', '_')
        for motif_id in top_motif_ids[:N_MOTIFS_PER_CELL_STATE]:
            sign, pattern_idx = re.fullmatch(r'(pos|neg)_patterns_pattern_(\d+)', motif_id).groups()
            clustering_id = f'{cell_state_idx}_{cell_state_name}_{sign}_{pattern_idx}'
            pwms[clustering_id] = cell_state_pwms[motif_id]
            nsites[clustering_id] = seqlet_counts[motif_id]
    motif.save_meme(pwms, input_meme_path, nsites=nsites)
    # matrix-clustering is run from the clustering directory, so paths are relative to it
    input_table_path.write_text(f'{input_meme_path.name}\t{COLLECTION_NAME}\tmeme\n')
    print(f"Saved {len(pwms)} motifs to cluster to {input_meme_path}.")

    # Clustering
    if not clusters_tab_path.exists():
        if args.matrix_clustering is None:
            matrix_clustering_path = 'path/to/matrix-clustering.R'
        else:
            matrix_clustering_path = str(args.matrix_clustering.resolve())
        command = [
            'Rscript', matrix_clustering_path,
            '-i', input_table_path.name,
            '-o', results_prefix.name,
            '-w', str(args.n_threads),
            '--export_heatmap', 'TRUE',
            '--linkage_method', LINKAGE,
            '--Ncor_th', str(NCOR_THRESHOLD),
        ]
        if args.matrix_clustering is None:
            print(f"matrix-clustering results not found. Run matrix-clustering from {args.clustering_dir} with:")
            print(' '.join(command))
            raise SystemExit
        subprocess.run(command, check=True, cwd=args.clustering_dir)

    # Cluster of each motif, from IDs with format '{collection}_{motif_id}_n{nsites}'
    clusters_df = pandas.read_csv(clusters_tab_path, sep='\t')
    motif_cluster_rows = []
    for cluster, motif_ids in zip(clusters_df['cluster'], clusters_df['id']):
        for motif_id in motif_ids.split(','):
            cell_state_idx, sign, pattern_idx = re.fullmatch(
                rf'{COLLECTION_NAME}_(\d+)_.*_(pos|neg)_(\d+)_n\d+',
                motif_id,
            ).groups()
            motif_cluster_rows.append((
                definitions.CELL_STATES[int(cell_state_idx)],
                f'{sign}_patterns_pattern_{pattern_idx}',
                cluster,
            ))
    motif_clusters_df = pandas.DataFrame(motif_cluster_rows, columns=['cell_state', 'motif', 'cluster'])
    motif_clusters_df.to_csv(args.clustering_dir / definitions.MOTIF_CLUSTERS_PATH.name, sep='\t', index=False)

    # Consensus PWM of each cluster
    with open(root_motifs_path) as f:
        root_motifs = bio_motifs.parse(f, 'transfac', strict=False)
    cluster_pwms = {}
    cluster_nsites = {}
    for root_motif in root_motifs:
        counts = numpy.array([root_motif.counts[base] for base in 'ACGT']).T
        cluster_pwms[root_motif['AC']] = counts / counts.sum(axis=1, keepdims=True)
        cluster_nsites[root_motif['AC']] = int(counts[0].sum())
    motif.save_meme(cluster_pwms, args.clustering_dir / definitions.CLUSTER_PWMS_PATH.name, nsites=cluster_nsites)
    print(
        f"{len(motif_clusters_df)} motifs in {len(cluster_pwms)} clusters, saved to {args.clustering_dir}."
    )
