import shutil
import urllib.request

# TODO: replace with the actual repository (e.g. Zenodo record) once files are uploaded
BASE_URL = 'https://example.org/deepdanio'


def download_files(url_path_list):
    """
    Download files, skipping those already present.

    Each file is downloaded to a temporary file first, so that interrupted
    downloads are not mistaken for complete ones.

    Parameters
    ----------
    url_path_list : list of (str, pathlib.Path)
        URLs and destination paths.

    """
    for url, dest_path in url_path_list:
        if dest_path.exists():
            print(f"{dest_path} already exists, skipping.")
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        tmp_path = dest_path.with_name(dest_path.name + '.part')
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp_path, 'wb') as f:
            shutil.copyfileobj(response, f)
        tmp_path.rename(dest_path)
