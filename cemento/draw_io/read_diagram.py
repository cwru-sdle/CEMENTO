from collections.abc import Iterable
from itertools import chain
from pathlib import Path

from more_itertools import unique_everseen

from cemento.draw_io.constants import BadDiagramError
from cemento.draw_io.io import write_error_diagram
from cemento.draw_io.preprocessing import (
    find_errors_diagram_content,
    get_diagram_error_exemptions,
)
from cemento.draw_io.transforms import (
    extract_elements,
    get_container_values,
    parse_containers,
    parse_elements,
)


def read_drawio(input_path: str | Path, check_errors: bool = False) -> tuple[
    dict[str, dict[str, any]],
    Iterable[str],
    list[tuple[str, str, str]],
    list[tuple[str, str, list[str]]],
]:

    elements = parse_elements(input_path)
    containers = parse_containers(elements)
    container_labels = get_container_values(containers, elements)
    non_container_elements = dict(
        filter(lambda item: item[0] not in containers.keys(), elements.items())
    )
    term_ids, property_ids = extract_elements(non_container_elements)

    error_exemptions = get_diagram_error_exemptions(non_container_elements)

    if check_errors:
        print("Checking for diagram errors...")
        container_content = set(chain(*containers.values()))
        errors = find_errors_diagram_content(
            elements,
            term_ids,
            property_ids,
            serious_only=True,
            containers=containers,
            container_content=container_content,
            error_exemptions=error_exemptions,
        )
        if errors:
            checked_diagram_path = write_error_diagram(input_path, errors)
            print(
                "The inputted file came down with the following problems. Please fix them appropriately."
            )
            for elem_id, error in errors:
                print(elem_id, error)
            raise BadDiagramError(checked_diagram_path)

    output_containers = {
        key: (container_labels[key], containers[key]) for key in containers
    }
    triples = [
        (
            elements[triple_id]["source"],
            triple_id,
            elements[triple_id]["target"],
        )
        for triple_id in property_ids
    ]
    all_terms = unique_everseen(chain(term_ids, property_ids))
    return elements, all_terms, triples, output_containers
