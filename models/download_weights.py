"""
Download trained model weights.

Each group of files is downloaded after confirmation. Files already present
are skipped.

"""
from deepdanio import definitions, download

if __name__ == '__main__':

    # DeepDanio, one model per chromosome split
    ###########################################
    if input("Download DeepDanio model weights? (Y/N): ").strip().lower() == 'y':
        url_path_list = []
        for split, model_path in definitions.DEEPDANIO_MODEL_PATHS.items():
            url_path_list.append((f'{download.BASE_URL}/{model_path.name}', model_path))
        download.download_files(url_path_list)
