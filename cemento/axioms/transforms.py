import sys
from collections import defaultdict
from functools import partial
from itertools import chain, product, filterfalse
from pprint import pprint

import networkx as nx
import rdflib
from matplotlib import pyplot as plt
from more_itertools import partition
from more_itertools.more import map_reduce, filter_map
from more_itertools.recipes import flatten
from networkx import DiGraph
from rdflib import Graph, BNode, URIRef, Literal
from rdflib.collection import Collection
from rdflib.namespace import RDF, SKOS, OWL, RDFS
from thefuzz.process import extractOne

from cemento.axioms.constants import combinators, class_rest_preds, prop_rest_preds
from cemento.axioms.modules import MS, MDS
from cemento.draw_io.constants import DiagramKey
from cemento.rdf.filters import term_not_in_default_namespace, term_in_search_results
from cemento.rdf.io import get_diagram_terms_iter
from cemento.rdf.transforms import (
    construct_terms,
    get_collection_nodes,
    get_collection_in_edges,
    split_collection_graph,
    get_search_keys,
    get_graph_diagram_terms_with_pred,
    construct_literal_terms,
    get_literal_terms,
    get_collection_triples_and_targets,
    add_collection_links_to_graph,
    get_exact_match_properties,
    get_ref_graph,
)
from cemento.term_matching.constants import get_default_namespace_prefixes
from cemento.term_matching.io import get_rdf_file_iter
from cemento.term_matching.transforms import (
    substitute_term,
    get_search_terms,
    get_prefixes,
)
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_reserved_references_folder,
    get_default_references_folder,
)
from cemento.utils.utils import (
    get_graph_root_nodes,
    get_subgraphs,
    chain_filter,
    get_abbrev_term,
    get_uri_ref_str,
)


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
    restriction_container_ids.update(base_restriction_box_ids)
    return base_restriction_box_ids, restriction_container_ids


def split_restriction_graph(
    graph: DiGraph,
    containers: dict[str, list[str]],
    container_labels: dict[str, str],
    base_restriction_box_ids: set[str],
    restriction_container_ids: set[str],
):
    element_containers, restriction_containers = partition(
        lambda item: item[0] in restriction_container_ids, containers.items()
    )
    element_containers = dict(element_containers)
    restriction_containers = dict(restriction_containers)
    collection_edges = list(
        flatten(map(lambda item: list(product([item[0]], item[1])), containers.items()))
    )
    graph.add_edges_from(collection_edges, label="mds:hasCollectionMember")
    graph.remove_nodes_from(base_restriction_box_ids)
    graph = get_container_collection_types(graph, container_labels, element_containers)
    restriction_nodes = filter(
        lambda node: "parent" in node[1]
        and node[1]["parent"] in restriction_container_ids,
        graph.nodes(data=True),
    )
    restriction_nodes = set(map(lambda node: node[0], restriction_nodes))
    restriction_nodes.update(restriction_container_ids)
    base_graph = graph.copy()
    restriction_graph = graph.subgraph(restriction_nodes).copy()
    restriction_containers = {
        key: value
        for key, value in restriction_containers.items()
        if key not in base_restriction_box_ids
    }
    restriction_graph = get_container_collection_types(
        restriction_graph, container_labels, restriction_containers
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
    return graph, restriction_graph


def convert_axiom_graph_to_rdf(
    graph: DiGraph,
    onto_ref_folder,
    defaults_folder,
    prefixes_path,
    log_substitution_path,
) -> Graph:
    onto_ref_folder = (
        get_default_references_folder() if not onto_ref_folder else onto_ref_folder
    )
    defaults_folder = (
        get_default_defaults_folder() if not defaults_folder else defaults_folder
    )
    prefixes_path = get_default_prefixes_file() if not prefixes_path else prefixes_path
    prefixes, inv_prefixes = get_prefixes(prefixes_path, onto_ref_folder)
    collection_nodes = get_collection_nodes(graph)
    collection_in_edges = get_collection_in_edges(collection_nodes.keys(), graph)
    collection_in_edge_labels = list(
        map(
            lambda x: x[2]["label"],
            filter(lambda x: "label" in x[2], collection_in_edges),
        )
    )
    graph, collection_subgraph = split_collection_graph(graph, collection_nodes)
    diagram_term_names = map(
        lambda item: (
            item,
            item if item not in graph.nodes else graph.nodes[item].get("label", item),
        ),
        get_diagram_terms_iter(graph),
    )
    diagram_term_names = filter(lambda item: item[1] is not None, diagram_term_names)
    diagram_term_names = chain(
        diagram_term_names, map(lambda term: (term, term), collection_in_edge_labels)
    )
    diagram_term_names = dict(diagram_term_names)
    all_diagram_terms = list(
        chain(diagram_term_names.values(), collection_in_edge_labels)
    )
    search_keys = get_search_keys(all_diagram_terms, inv_prefixes)
    search_terms = get_search_terms(
        inv_prefixes,
        onto_ref_folder,
        defaults_folder,
        extra_paths=[get_reserved_references_folder()],
    )
    graph_edge_labels = set(
        map(lambda edge_tuple: edge_tuple[2]["label"], graph.edges(data=True))
    )
    graph_edge_labels |= set(collection_in_edge_labels)
    all_diagram_terms_with_pred = list(
        map(lambda term: (term, term in graph_edge_labels), all_diagram_terms)
    )
    constructed_terms = construct_terms(
        all_diagram_terms,
        all_diagram_terms_with_pred,
        prefixes,
        search_keys,
        search_terms,
        prefixes_path,
        log_substitution_path,
        default_prefix_for_unassigned="ms",
    )

    expanded_axiom_rdf_graph = Graph()
    id_to_uri_mapping = {
        key: constructed_terms[value] for key, value in diagram_term_names.items()
    }
    uri_to_id_mapping = {value: key for key, value in id_to_uri_mapping.items()}
    collection_triples, collection_targets = get_collection_triples_and_targets(
        collection_nodes,
        collection_subgraph,
        expanded_axiom_rdf_graph,
        id_to_uri_mapping,
    )

    for triple in collection_triples:
        expanded_axiom_rdf_graph.add(triple)
    graph = add_collection_links_to_graph(
        collection_in_edges, collection_targets, graph
    )

    output_graph = nx.DiGraph()
    for subj, obj, data in graph.edges(data=True):
        pred = data["label"]
        output_graph.add_edge(subj, obj, label=pred)

    # set partial for converting URIRef to string
    uri_to_str = partial(get_uri_ref_str, inv_prefixes=inv_prefixes)

    # get intro terms before deleting the triples
    intro_terms = list(
        map(
            lambda edge: edge[0],
            filter(
                lambda edge: edge[1] in id_to_uri_mapping
                and id_to_uri_mapping[edge[1]] == MS.IntroTerm,
                graph.edges,
            ),
        )
    )

    edge_labels_to_remove = [MS.belongsTo, SKOS.exactMatch, RDF.type]
    edge_labels_to_remove = set(map(uri_to_str, edge_labels_to_remove))
    edges_to_remove = list(
        map(
            lambda edge: (edge[0], edge[1]),
            filter(
                lambda edge: edge[2].get("label", None) in edge_labels_to_remove,
                graph.edges(data=True),
            ),
        )
    )
    graph.remove_edges_from(edges_to_remove)

    # TODO: add mapping to exact matches in namespace class generator script
    ms_ttl_term_mapping = {
        MS.equivalentTo: OWL.equivalentClass,
        MS.some: OWL.someValuesFrom,
        MS.of: OWL.onClass,
        MS.max: OWL.maxQualifiedCardinality,
        MS.And: OWL.intersectionOf,
        MS.Or: OWL.unionOf,
        MS.min: OWL.minQualifiedCardinality,
        MS.only: OWL.allValuesFrom,
    }

    # initiate chain
    # NOTE: the chains only apply to property restrictions!
    # TODO: add support for just datatype. Datatype facets for property restrictions are supported.
    expanded_axiom_rdf_graph = rdflib.Graph()
    for prefix, namespace_uri in prefixes.items():
        expanded_axiom_rdf_graph.bind(prefix, namespace_uri)

    pivot_nodes = list(
        filter(
            lambda node: id_to_uri_mapping.get(node, None) in combinators, graph.nodes
        )
    )
    pivot_node_predicates = {
        pivot_node: list(
            map(
                lambda edge: id_to_uri_mapping[edge[2].get("label", edge[2])],
                graph.out_edges(pivot_node, data=True),
            )
        )
        for pivot_node in pivot_nodes
    }
    # TODO: check that the restriction graph has no floating nodes
    # TODO: check if the predicates are all the same type as a preprocessing step
    pivots_with_types = filter(
        lambda item: item[1][0] in (class_rest_preds | prop_rest_preds),
        pivot_node_predicates.items(),
    )
    prop_pivots, class_pivots = partition(
        lambda item: item[1][0] in class_rest_preds, pivots_with_types
    )
    prop_pivots = map(lambda item: (item[0], "prop"), prop_pivots)
    class_pivots = map(lambda item: (item[0], "class"), class_pivots)
    pivot_node_types = dict(chain(class_pivots, prop_pivots))
    compressed_graph = nx.DiGraph()
    node_containers = defaultdict(list)
    pivot_subjects = dict()
    # FIXME: adjust algorithm to work for simple graphs (graphs without AND or OR)
    for intro_term in intro_terms:
        combinator_parents = dict()
        current_pivot = None
        current_parent = None
        current_node = None
        starting = True
        for subj, obj in nx.dfs_edges(graph, source=intro_term):
            pred = graph[subj][obj].get("label", None)
            if starting:
                starting_node = subj
                node_containers[starting_node].append((pred, subj))
                current_node = starting_node
                starting = False
            if obj in pivot_nodes or (subj in pivot_nodes and current_pivot != subj):
                if subj in pivot_nodes:
                    parent_node = combinator_parents[subj]
                else:
                    parent_node = subj
                    combinator_parents[obj] = parent_node
                    compressed_graph.add_node(
                        parent_node,
                        subject=subj,
                        combinator=obj,
                        type=pivot_node_types[obj],
                    )
                current_pivot = obj
                pivot_subjects[obj] = parent_node
                current_parent = parent_node
                current_node = None
            if obj not in pivot_nodes and subj in pivot_nodes:
                current_node = obj
                node_containers[current_node].append((pred, obj))
                compressed_graph.add_edge(current_parent, current_node)
            if obj not in pivot_nodes and subj not in pivot_nodes:
                node_containers[current_node].append((pred, obj))

    node_bnode_mapping = {node: BNode() for node in compressed_graph.nodes}
    compressed_graph = nx.relabel_nodes(compressed_graph, node_bnode_mapping)
    node_containers = {
        node_bnode_mapping[key]: values for key, values in node_containers.items()
    }

    trees = get_subgraphs(compressed_graph)
    for tree in trees:
        terms_to_unwrap = dict()
        for node in nx.dfs_postorder_nodes(tree):
            node_data = tree.nodes[node]
            print(node)
            if node_data:  # if there is data, it is a combinator
                pivot_type = node_data["type"]
                pivot_subject = id_to_uri_mapping[node_data["subject"]]
                pivot_combinator = ms_ttl_term_mapping[
                    id_to_uri_mapping[node_data["combinator"]]
                ]
                members = list(tree.successors(node))
                inner_node = BNode()
                expanded_axiom_rdf_graph.add((node, pivot_combinator, inner_node))
                if pivot_type == "prop":
                    for member in members:
                        expanded_axiom_rdf_graph.add(
                            (member, OWL.onProperty, pivot_subject)
                        )
                elif pivot_type == "class":
                    # assume all members have the same predicate to start
                    conn_pred = id_to_uri_mapping[node_containers[members[0]][0][0]]
                    expanded_axiom_rdf_graph.add((pivot_subject, conn_pred, node))

                # unwrap singular members and add the unwrapped node to the collection
                members = set(tree.successors(node))
                members_to_unwrap = members & terms_to_unwrap.keys()
                members_to_unwrap = set(
                    map(
                        lambda node: id_to_uri_mapping[terms_to_unwrap[node]],
                        members_to_unwrap,
                    )
                )
                rem_members = members - terms_to_unwrap.keys()
                members = rem_members | members_to_unwrap
                Collection(expanded_axiom_rdf_graph, inner_node, members)  # STARBOY

            else:  # else, it is a branch
                if len(node_containers[node]) == 1:
                    pred, obj = node_containers[node][0]
                    if id_to_uri_mapping[pred] in class_rest_preds:
                        terms_to_unwrap[node] = obj
                        continue
                expanded_axiom_rdf_graph.add((node, RDF.type, OWL.Restriction))
                for pred, obj in node_containers[node]:
                    pred = id_to_uri_mapping.get(pred, pred)
                    obj = id_to_uri_mapping.get(obj, obj)
                    if pred in class_rest_preds:
                        pred = OWL.onProperty
                    if pred in ms_ttl_term_mapping:
                        pred = ms_ttl_term_mapping[pred]
                    expanded_axiom_rdf_graph.add((node, pred, obj))

            pprint(node_containers[node])
            print()
        print("---" * 10)

    expanded_axiom_rdf_graph.serialize("axiom_intermediate.ttl", format="turtle")

    return expanded_axiom_rdf_graph
