from pathlib import Path

from cemento.draw_io.read_diagram import read_drawio
from cemento.rdf.graph_to_rdf import convert_graph_to_rdf_file
from cemento.utils.constants import RDFFormat


def convert_drawio_to_rdf(
    input_path: str | Path,
    output_path: str | Path,
    file_format: str | RDFFormat = None,
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    prefixes_path: str | Path = None,
    check_errors: bool = False,
    log_substitution_path: str | Path = None,
) -> None:
    elements, all_terms, triples = read_drawio(
        input_path,
        check_errors=check_errors,
    )
    convert_graph_to_rdf_file(
        elements,
        all_terms,
        triples,
        output_path,
        file_format=file_format,
        onto_ref_folder=onto_ref_folder,
        defaults_folder=defaults_folder,
        prefixes_path=prefixes_path,
        log_substitution_path=log_substitution_path,
    )
