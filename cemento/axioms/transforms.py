from networkx import DiGraph
import networkx as nx
from cemento.draw_io.constants import DiagramKey
from collections.abc import Container
from more_itertools import partition
from cemento.term_matching.transforms import substitute_term
from itertools import chain
from functools import partial
from cemento.utils.utils import get_graph_root_nodes


def relabel_graph_nodes_with_node_attr(
    graph: DiGraph, new_attr_label: str = DiagramKey.TERM_ID.value
) -> DiGraph:
    node_info = nx.get_node_attributes(graph, new_attr_label)
    relabel_mapping = {
        current_node_label: (
            node_info[current_node_label]
            if current_node_label in node_info
            else current_node_label
        )
        for current_node_label in graph.nodes
    }
    return nx.relabel_nodes(graph, relabel_mapping)


def link_container_members(graph: DiGraph, containers: dict[str, list[str]]) -> DiGraph:
    graph = graph.copy()
    for container_id, items in containers.items():
        for item in items:
            graph.add_edge(container_id, item, label="mds:hasCollectionMember")
    return graph


def get_container_collection_types(
    graph: DiGraph, container_labels: dict[str, str], containers: dict[str, list[str]]
) -> dict[str, str]:
    graph = graph.copy()
    for container_id in containers.keys():
        container_type = container_labels[container_id]
        if not container_type.strip():
            container_type = "mds:tripleSyntaxSugar"
        else:
            # TODO: move search terms to constants file
            valid_collection_types = {
                "owl:unionOf",
                "owl:intersectionOf",
                "owl:complementOf",
                "mds:tripleSyntaxSugar",
            }
            container_type, did_substitute = substitute_term(
                container_type,
                valid_collection_types,
            )
            # TODO: move to error check
            if not did_substitute:
                raise ValueError(
                    f"The provided collection header does not seem to match any of the valid collection types. Choose between: {valid_collection_types} or leaving the header blank."
                )
        graph.add_edge(container_type, container_id)
    return graph


def split_container_ids(
    container_labels: dict[str, str],
    containers: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    # separate container IDs between element and restriction boxes
    # TODO: fuzzy match for owl:Restriction
    base_restriction_box_ids = set(
        filter(
            lambda container_id: container_labels[container_id] == "owl:Restriction",
            containers,
        )
    )
    restriction_box_content_ids = chain.from_iterable(
        map(lambda box_id: containers[box_id], base_restriction_box_ids)
    )
    restriction_box_ids = set(
        chain(base_restriction_box_ids, restriction_box_content_ids)
    )
    restriction_container_ids = filter(
        lambda container_id: container_id in restriction_box_ids, containers
    )
    restriction_container_ids = set(restriction_container_ids)
    return base_restriction_box_ids, restriction_container_ids


def split_restriction_graph(
    graph: DiGraph,
    containers: dict[str, list[str]],
    container_labels: dict[str, str],
    base_restriction_box_ids: set[str],
    restriction_container_ids: Container[str],
    relabel_key: DiagramKey = DiagramKey.LABEL,
):
    element_containers, restriction_containers = partition(
        lambda item: item[0] in restriction_container_ids, containers.items()
    )
    element_containers = dict(element_containers)
    restriction_containers = dict(restriction_containers)

    graph = get_container_collection_types(graph, container_labels, element_containers)
    graph = link_container_members(graph, element_containers)
    restriction_nodes = filter(
        lambda node: "parent" in node[1]
        and node[1]["parent"] in restriction_container_ids,
        graph.nodes(data=True),
    )
    restriction_nodes = list(map(lambda node: node[0], restriction_nodes))
    base_graph = graph.copy()
    restriction_graph = graph.subgraph(restriction_nodes).copy()
    restriction_graph.remove_nodes_from(base_restriction_box_ids)
    restriction_containers = {
        key: value
        for key, value in restriction_containers.items()
        if key not in base_restriction_box_ids
    }
    restriction_graph = get_container_collection_types(
        restriction_graph, container_labels, restriction_containers
    )
    restriction_graph = link_container_members(
        restriction_graph, restriction_containers
    )
    graph.remove_nodes_from(restriction_nodes)
    restriction_graph_roots = get_graph_root_nodes(restriction_graph)
    restriction_in_edges = list(
        chain.from_iterable(
            map(
                lambda root: base_graph.in_edges(root, data=True),
                restriction_graph_roots,
            )
        )
    )
    restriction_in_edge_nodes = list(
        chain.from_iterable(
            map(lambda nodes: (nodes[0], nodes[1]), restriction_in_edges)
        )
    )
    restriction_graph.add_nodes_from(
        base_graph.subgraph(restriction_in_edge_nodes).nodes(data=True)
    )
    restriction_graph.add_edges_from(restriction_in_edges)
    relabel_graph = partial(
        relabel_graph_nodes_with_node_attr, new_attr_label=relabel_key.value
    )
    graph, restriction_graph = tuple(map(relabel_graph, (graph, restriction_graph)))

    return graph, restriction_graph
