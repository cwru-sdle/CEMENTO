import html
import re
from collections.abc import Container, Iterable

import networkx as nx
from bs4 import BeautifulSoup
from networkx import DiGraph

from cemento.draw_io.constants import (
    BidirectionalEdgeError,
    BlankEdgeLabelError,
    BlankTermLabelError,
    CircularEdgeError,
    Connector,
    ContainerSubjectError,
    DisconnectedTermError,
    FloatingContainerError,
    FloatingEdgeError,
    InvertedEdgeError,
    MissingChildEdgeError,
    MissingParentEdgeError,
    NestedSyntaxSugarError,
    NxEdge,
    Shape,
)
from cemento.utils.utils import fst, snd


def remove_literal_id(literal_content: str) -> str:
    # TODO: replace with hashed id pattern
    return re.sub(r"literal_id-(\w+):", "", literal_content)


def replace_term_quotes(graph: DiGraph) -> DiGraph:
    replace_nodes = {term: replace_quotes(term) for term in graph.nodes}
    return nx.relabel_nodes(graph, replace_nodes, copy=True)


def remove_predicate_quotes(edges: Iterable[NxEdge]) -> Iterable[NxEdge]:
    return map(
        lambda edge: (
            (edge.subj, edge.obj, remove_quotes(edge.pred)) if edge.pred else None
        ),
        edges,
    )


def escape_shape_content(shape: Shape) -> Shape:
    # TODO: implement immutable object copy
    shape.shape_content = html.escape(shape.shape_content, quote=True)
    return shape


def remove_literal_shape_id(shape: Shape) -> Shape:
    # TODO: implement immutable object copy
    shape.shape_content = remove_literal_id(shape.shape_content)
    return shape


def remove_literal_connector_id(connector: Connector) -> Connector:
    connector.connector_val = remove_literal_id(connector.connector_val)
    return connector


def clean_term_preserving_quotes(term: str) -> str:
    new_value = term.replace("&quot;", "{QUOTE_PLACEHOLDER}")
    new_value = clean_term(new_value)
    new_value = new_value.replace("{QUOTE_PLACEHOLDER}", "&quot;")
    return new_value


def clean_term(term: str) -> str:
    while (reduced_term := html.unescape(term).strip()) != term:
        term = reduced_term
    soup = BeautifulSoup(term, "html.parser")
    term_text = soup.get_text(separator="", strip=True)
    return term_text


def replace_quotes(input_str: str) -> str:
    return input_str.replace('"', "&quot;")


def remove_html_quote(input_str: str) -> str:
    return input_str.replace("&quot;", "")


def remove_quotes(input_str: str) -> str:
    if not input_str or not isinstance(input_str, str):
        return input_str
    return remove_html_quote(input_str.replace('"', "").strip())


def is_line(element: dict[str, any]) -> bool:
    return ("endArrow" in element and element["endArrow"].lower() == "none") and (
        "startArrow" not in element or element["startArrow"].lower() == "none"
    )


def get_diagram_error_exemptions(elements: dict[str, dict[str, any]]) -> set[str]:
    edges = {
        key: value
        for key, value in elements.items()
        if "edge" in value and value["edge"] == "1"
    }
    term_ids = elements.keys() - edges.keys()
    terms = {key: value for key, value in elements.items() if key in term_ids}

    lines = set(map(fst, filter(lambda edge: is_line(snd(edge)), edges.items())))

    # TODO: move to constants file
    reserved_term_annotations = {"t-box", "a-box"}

    reserved_terms = {
        term_id
        for term_id, term_element in terms.items()
        for reserved_term in reserved_term_annotations
        if "value" in term_element
        and reserved_term in term_element["value"].strip().lower()
    }

    return lines | reserved_terms


def get_connected_term_error_message(element_id, elements):
    error_message = None

    if not element_id:
        return error_message

    term = elements.get(element_id, None)
    value = term.get("value", None) if term else None
    if term and value:
        error_message = f"{value} ({element_id})"
    else:
        error_message = f"container with {element_id}"

    error_message = f"{error_message} located in ({term.get('x', 'Unknown')}, {term.get('y', 'Unknown')})"
    return error_message


def find_edge_errors_diagram_content(
    elements: dict[str, dict[str, any]],
    serious_only: bool = False,
) -> list[tuple[str, BaseException]]:
    edges = {
        key: value
        for key, value in elements.items()
        if "edge" in value and value["edge"] == "1"
    }

    errors = []

    for edge_id, edge_attr in edges.items():

        source_id = edge_attr.get("source", None)
        target_id = edge_attr.get("target", None)
        connected_terms = {
            get_connected_term_error_message(source_id, elements),
            get_connected_term_error_message(target_id, elements),
        } - {None, ""}

        edge_content = edge_attr.get("value", None)

        if "value" not in edge_attr or not edge_attr["value"]:
            errors.append((edge_id, BlankEdgeLabelError(edge_id, connected_terms)))

        if (
            ("startArrow" in edge_attr and edge_attr["startArrow"].lower() != "none")
            and ("endArrow" in edge_attr and edge_attr["endArrow"].lower() != "none")
            or (
                (
                    "startArrow" in edge_attr
                    and edge_attr["startArrow"].lower() != "none"
                )
                and "endArrow" not in edge_attr
            )
        ):
            connected_terms_iter = iter(connected_terms)
            errors.append(
                (
                    edge_id,
                    BidirectionalEdgeError(
                        edge_id,
                        edge_content,
                        next(connected_terms_iter, None),
                        next(connected_terms_iter, None),
                    ),
                )
            )
        elif (
            "startArrow" in edge_attr and edge_attr["startArrow"].lower() != "none"
        ) and ("endArrow" in edge_attr and edge_attr["endArrow"].lower() == "none"):
            connected_terms_iter = iter(connected_terms)
            errors.append(
                (
                    edge_id,
                    InvertedEdgeError(
                        edge_id,
                        edge_content,
                        next(connected_terms_iter, None),
                        next(connected_terms_iter, None),
                    ),
                )
            )

        if all(
            [
                "source" not in edge_attr or not edge_attr["source"],
                "target" not in edge_attr or not edge_attr["target"],
            ]
        ):
            errors.append((edge_id, FloatingEdgeError(edge_id, edge_content)))
            continue

        if "source" not in edge_attr or not edge_attr["source"]:
            errors.append(
                (
                    edge_id,
                    MissingParentEdgeError(
                        edge_id, edge_content, next(iter(connected_terms))
                    ),
                )
            )
            continue

        if "target" not in edge_attr or not edge_attr["target"]:
            errors.append(
                (
                    edge_id,
                    MissingChildEdgeError(
                        edge_id, edge_content, next(iter(connected_terms))
                    ),
                )
            )
            continue

        if (
            "target" in edge_attr
            and "source" in edge_attr
            and edge_attr["target"] == edge_attr["source"]
        ):
            errors.append((edge_id, CircularEdgeError(edge_id, edge_content)))

    if serious_only:
        # hide the errors related to circular edges (false positive bug with draw.io)
        circular_edge_errors = filter(
            lambda error: isinstance(snd(error), CircularEdgeError), errors
        )
        non_affected_ids = {id for id, error in circular_edge_errors}
        errors = list(filter(lambda error: fst(error) not in non_affected_ids, errors))

    return errors


# TODO: memoize
def get_connected_terms(elements: dict[str, dict[str, any]], rel_ids: set[str]):
    return {
        term
        for rel_id in rel_ids
        for term in (
            elements[rel_id].get("source", None),
            elements[rel_id].get("target", None),
        )
    } - {None, ""}


def find_shape_errors_diagram_content(
    elements: dict[str, dict[str, any]],
    term_ids: set[str],
    rel_ids: set[str],
    container_content: Container[str] = None,
) -> list[tuple[str, BaseException]]:
    connected_terms = get_connected_terms(elements, rel_ids)

    errors = []
    for term_id in term_ids:
        term = elements[term_id]
        if term_id not in connected_terms:
            if container_content is None or term_id not in container_content:
                errors.append((term_id, DisconnectedTermError(term_id, term["value"])))

        if "value" not in term or not term["value"]:
            errors.append((term_id, BlankTermLabelError(term_id)))
    return errors


def map_element_values(
    elements: dict[str, dict[str, any]], element_ids: Iterable[str]
) -> Iterable[str]:
    return (
        (
            element_attr["value"]
            if "value" in (element_attr := elements[element_id])
            else None
        )
        for element_id in element_ids
    )


def find_container_errors_diagram_content(
    elements: dict[str, dict[str, any]],
    containers: dict[str, list[str]],
    restriction_container_ids: Container[str],
    rel_ids: set[str],
) -> list[tuple[str, BaseException]]:
    errors = []
    if containers is None:
        return errors

    for container_id, members in containers.items():
        for member_id in members:
            member = elements[member_id].get("value", None)
            if member is None or not member.strip():
                errors.append((container_id, NestedSyntaxSugarError))

    collection_subject_rels = list(
        filter(
            lambda x: "source" in elements[x] and elements[x]["source"] in containers,
            rel_ids,
        )
    )
    for rel_id in collection_subject_rels:
        container_id = elements[rel_id]["source"]
        errors.append((container_id, ContainerSubjectError))

    connected_containers = set()
    for rel_id in rel_ids:
        source = elements[rel_id].get("source", None)
        target = elements[rel_id].get("target", None)
        if source in containers or target in containers:
            connected_containers.add(source)
            connected_containers.add(target)

    connected_containers -= {None, ""}

    nested_containers = set()
    for members in containers.values():
        for member in members:
            if member in containers:
                nested_containers.add(member)
    floating_containers = (
        set(containers.keys()) - connected_containers - nested_containers
    )
    floating_containers = filter(lambda container_id: container_id not in restriction_container_ids, floating_containers)

    for container_id in floating_containers:
        errors.append((container_id, FloatingContainerError))

    submit_errors = []
    for container_id, ErrorType in errors:
        members = list(
            map(
                lambda member_id: (member_id, elements[member_id].get("value", None)),
                containers[container_id],
            )
        )
        member_ids, member_values = zip(*members, strict=True)
        container_value = elements[container_id].get("value", None)
        submit_errors.append(
            (
                container_id,
                ErrorType(
                    container_id, container_value, list(member_ids), list(member_values)
                ),
            )
        )

    return submit_errors


def find_errors_diagram_content(
    elements: dict[str, dict[str, any]],
    term_ids: set[str],
    rel_ids: set[str],
    serious_only: bool = False,
    containers: dict[str, list[str]] = None,
    container_content: Container[str] = None,
    restriction_container_ids: Container[str] = None,
    error_exemptions: set[str] = None,
) -> list[tuple[str, BaseException]]:
    errors = (
        find_shape_errors_diagram_content(
            elements, term_ids, rel_ids, container_content
        )
        + find_edge_errors_diagram_content(elements, serious_only=serious_only)
        + find_container_errors_diagram_content(elements, containers, restriction_container_ids, rel_ids)
    )
    if error_exemptions is not None:
        errors = list(
            filter(lambda error_info: fst(error_info) not in error_exemptions, errors)
        )
    return errors
