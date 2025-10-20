from importlib import resources
from pathlib import Path

from cemento.utils.constants import RDFFormat


def get_default_path(rel_path: str | Path) -> Path:
    try:
        return resources.files("cemento.data") / rel_path
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        return Path(__file__).parent / "data" / rel_path


def get_default_defaults_folder() -> Path:
    return get_default_path("defaults")


def get_default_references_folder() -> Path:
    return get_default_path("references")


def get_default_reserved_folder() -> Path:
    return get_default_path("reserved")


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
