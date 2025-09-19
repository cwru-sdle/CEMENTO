from functools import partial
from itertools import chain
from pathlib import Path
import sys
from pprint import pprint

from more_itertools import partition
from networkx import DiGraph

from cemento.axioms.transforms import (
    split_restriction_graph,
    split_container_ids,
    relabel_graph_nodes_with_node_attr,
)
from cemento.draw_io.constants import BadDiagramError, DiagramKey
from cemento.draw_io.io import write_error_diagram
from cemento.draw_io.preprocessing import (
    find_errors_diagram_content,
    get_diagram_error_exemptions,
)
from cemento.draw_io.transforms import (
    extract_elements,
    generate_graph,
    get_container_values,
    parse_containers,
    parse_elements,
)
from cemento.term_matching.transforms import get_prefixes, get_strat_predicates_str
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
)


def read_drawio(
    input_path: str | Path,
    onto_ref_folder: str | Path = None,
    prefixes_file: str | Path = None,
    defaults_folder: str | Path = None,
    relabel_key: DiagramKey = DiagramKey.LABEL,
    check_errors: bool = False,
    inverted_rank_arrow: bool = False,
) -> DiGraph:
    prefixes_file = get_default_prefixes_file() if not prefixes_file else prefixes_file
    defaults_folder = (
        get_default_defaults_folder() if not defaults_folder else defaults_folder
    )
    onto_ref_folder = (
        get_default_references_folder() if not onto_ref_folder else onto_ref_folder
    )

    elements = parse_elements(input_path)
    containers = parse_containers(elements)
    container_content = set(chain(*containers.values()))
    container_labels = get_container_values(containers, elements)
    non_container_elements = dict(
        filter(lambda item: item[0] not in containers.keys(), elements.items())
    )
    term_ids, rel_ids = extract_elements(non_container_elements)
    strat_props = None
    prefixes, inv_prefixes = get_prefixes(prefixes_file, onto_ref_folder)
    strat_props = get_strat_predicates_str(
        onto_ref_folder, defaults_folder, inv_prefixes
    )

    error_exemptions = get_diagram_error_exemptions(non_container_elements)

    base_restriction_box_ids, restriction_container_ids = split_container_ids(
        container_labels,
        containers,
    )

    if check_errors:
        print("Checking for diagram errors...")
        # TODO: write custom error checks specifically for axioms
        errors = find_errors_diagram_content(
            elements,
            term_ids,
            rel_ids,
            containers,
            container_content,
            restriction_container_ids,
            error_exemptions,
            serious_only=True,
        )
        if errors:
            checked_diagram_path = write_error_diagram(input_path, errors)
            print(
                "The inputted file came down with the following problems. Please fix them appropriately."
            )
            for elem_id, error in errors:
                print(elem_id, error)
            raise BadDiagramError(checked_diagram_path)

    print("generating graph...")
    graph = generate_graph(
        non_container_elements,
        term_ids,
        rel_ids,
        strat_terms=strat_props,
        exempted_elements=error_exemptions,
        inverted_rank_arrow=inverted_rank_arrow,
    )
    graph, restriction_graph = split_restriction_graph(
        graph,
        containers,
        container_labels,
        base_restriction_box_ids,
        restriction_container_ids,
    )
    graph = relabel_graph_nodes_with_node_attr(graph, new_attr_label=relabel_key.value)
    return graph, restriction_graph
