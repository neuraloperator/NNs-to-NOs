from pathlib import Path
from neuralop.data.datasets.web_utils import download_from_zenodo_record

if __name__ == '__main__':
    # Define the target directory
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo record ID
    record_id = "15687518"

    # List of files to download from the record
    files_to_download = ["nsforcing_1024_test1.pt", 
                    "nsforcing_128_test.pt",
                    "nsforcing_128_train.pt",
                    "nsforcing_256_test.pt",
                    "nsforcing_256_train.pt",
                    "nsforcing_64_test.pt",
                    "nsforcing_64_train.pt"]

    # Download files
    download_from_zenodo_record(
        record_id=record_id,
        root=data_dir,
        files_to_download=files_to_download
    )