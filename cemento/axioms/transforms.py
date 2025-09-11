from pprint import pprint
import sys

from more_itertools.recipes import flatten
from rdflib import Graph
from rdflib.namespace import RDF, SKOS
from networkx import DiGraph
import networkx as nx
from cemento.draw_io.constants import DiagramKey
from collections.abc import Container
from more_itertools import partition
from cemento.term_matching.transforms import substitute_term
from itertools import chain
from functools import partial
from cemento.utils.utils import get_graph_root_nodes
from cemento.axioms.modules import MS


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
    restriction_container_graph = nx.DiGraph()
    restriction_container_graph.add_edges_from(
        {(key, value) for key, values in containers.items() for value in values}
    )
    restriction_container_ids = set(
        flatten(
            map(
                lambda box_id: nx.descendants(restriction_container_graph, box_id),
                base_restriction_box_ids,
            )
        )
    )
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
    print(restriction_container_ids)
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
    for subj, _, _ in restriction_in_edges:
        restriction_graph.add_edge(subj, "ms:introTerm", label="ms:belongsTo")

    relabel_graph = partial(
        relabel_graph_nodes_with_node_attr, new_attr_label=relabel_key.value
    )
    graph, restriction_graph = tuple(map(relabel_graph, (graph, restriction_graph)))

    return graph, restriction_graph


def expand_axiom_terms(restriction_rdf_graph: Graph) -> Graph:
    graph = nx.DiGraph()
    intro_terms = list(restriction_rdf_graph.subjects(MS.belongsTo, MS.IntroTerm))
    # restriction_rdf_graph.remove((None, RDF.type, None))
    # restriction_rdf_graph.remove((None, SKOS.exactMatch, None))
    # restriction_rdf_graph.remove((None, MS.belongsTo, None))
    graph.add_edges_from(
        ((subj, obj, {"label": pred}) for subj, pred, obj in restriction_rdf_graph)
    )
    restriction_rdf_graph.serialize("intermediate.ttl", format="turtle")
    # # for term in intro_terms:
    # for subj, obj in nx.dfs_edges(graph):
    #     pred = graph[subj][obj].get('label', None)
    #     print(subj, pred, obj)

    return restriction_rdf_graph
