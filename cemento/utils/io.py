import os
from http.client import responses
from importlib import resources
from pathlib import Path

import requests

from cemento.utils.constants import RDFFormat, DEFAULT_DOWNLOADS, APP_NAME

from pathlib import Path
from platformdirs import user_data_dir


def make_data_dirs(download_path: str | Path) -> Path:
    download_path = Path(download_path)
    if not download_path.exists() or not download_path.is_dir():
        print("creating a download directory for default ontology reference files...")
        os.mkdir(download_path)
    return download_path


def download_default_reference_ontos(data_path: Path) -> list[str]:
    if not data_path.exists() or not data_path.is_dir():
        raise ValueError("The specified download folder does not exist!")

    download_files = []
    for key, url in DEFAULT_DOWNLOADS.items():
        default_file = data_path / f"{key}.ttl"
        if not default_file.exists():
            download_path = download_onto(url, default_file)
            if download_path is not None:
                download_files.append(str(default_file))
    return download_files


def download_onto(download_url: str, output_path: Path) -> Path | None:
    try:
        print(f"attempting to download {output_path.name} from {download_url}...")
        response = requests.get(download_url, timeout=10)
        response.raise_for_status()
        output_path.write_bytes(response.content)
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        print(f"Download failed for {output_path.name}....")
        return None
    return output_path


def update_reference_ontos():
    user_path = Path(user_data_dir(APP_NAME))
    pkg_default_refs_folder = get_default_path("references")
    default_refs_paths = {user_path / f"{filename}.ttl": url for filename, url in DEFAULT_DOWNLOADS.items()}
    for default_ref_file_path, url in default_refs_paths.items():
        if not default_ref_file_path.exists():
            print(f"Warning: The file {default_ref_file_path} was not found in resources.")
            # create directory if it doesn't exist
            user_path.mkdir(parents=True, exist_ok=True)
            # attempt to download first
            download_path = download_onto(url, default_ref_file_path)
            if download_path is None:
                print("File could not be downloaded! Will copy from packaged resources.")
                copy_path = pkg_default_refs_folder / default_ref_file_path.name
                default_ref_file_path.write_bytes(copy_path.read_bytes())


def get_default_path(rel_path: str | Path) -> Path:
    try:
        return resources.files("cemento.data") / rel_path
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        return Path(__file__).parent / "data" / rel_path


def get_default_defaults_folder() -> Path:
    return get_default_path("defaults")


def get_default_references_folder() -> Path:
    return Path(user_data_dir(APP_NAME))


def get_default_reserved_folder() -> Path:
    return get_default_path("reserved")


def get_data_folders() -> list[Path]:
    return [
        get_default_defaults_folder(),
        get_default_references_folder(),
        get_default_reserved_folder()
    ]


def get_default_prefixes_file() -> Path:
    return get_default_path("default_prefixes.json")


def get_rdf_format(file_path: str | Path, file_format: str | RDFFormat = None) -> str:
    file_path = Path(file_path)

    rdf_format = None
    if file_format is None:
        file_ext = file_path.suffix
        rdf_format = RDFFormat.from_ext(file_ext)
    elif isinstance(file_format, str):
        rdf_format = RDFFormat.from_input(file_format)

    rdf_format = (
        file_format
        if file_format is not None and isinstance(file_format, str)
        else rdf_format.value
    )
    return rdf_format
