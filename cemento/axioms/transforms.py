import logging
import re
import sys
from collections import defaultdict
from pkgutil import extend_path
from pprint import pprint

import rdflib
from more_itertools.more import map_reduce
from thefuzz.process import extractOne
from more_itertools.recipes import flatten, unique_everseen
from rdflib import Graph, BNode, Literal, XSD
from rdflib.namespace import RDF, SKOS, OWL, RDFS
from networkx import DiGraph
import networkx as nx
from cemento.draw_io.constants import DiagramKey
from collections.abc import Container
from more_itertools import partition
from cemento.term_matching.transforms import substitute_term, get_term_search_keys
from itertools import chain, product, filterfalse
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
    relabel_key: DiagramKey = DiagramKey.LABEL,
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

    # TODO: move to own function
    repeated_nodes = defaultdict(int)
    combinator_term_add_edges = []
    for node, data in restriction_graph.nodes(data=True):
        label = data.get("label", None)
        # TODO: move refs to constants
        if (
            label is not None
            and label
            and (
                substitute_label := extractOne(
                    label, {"ms:and", "ms:or"}, score_cutoff=90
                )
            )
        ):
            substitute_label = substitute_label[0]
            substitute_label_value = substitute_label.replace("ms:", "")
            repeated_nodes[substitute_label] += 1
            idx = repeated_nodes[substitute_label]
            restriction_graph.nodes[node][
                "label"
            ] = f"ms:combinator-{substitute_label_value}-iter-{idx}"
            combinator_term_add_edges.append(
                (node, substitute_label, {"label": "skos:exactMatch"})
            )
    restriction_graph.add_edges_from(combinator_term_add_edges)
    relabel_graph = partial(
        relabel_graph_nodes_with_node_attr, new_attr_label=relabel_key.value
    )
    graph, restriction_graph = tuple(map(relabel_graph, (graph, restriction_graph)))

    return graph, restriction_graph


def expand_axiom_terms(restriction_rdf_graph: Graph) -> Graph:
    graph = nx.DiGraph()
    repeated_combinators = restriction_rdf_graph.triples_choices(
        (None, SKOS.exactMatch, [MS.And, MS.Or])
    )
    repeated_combinators = filter(
        lambda triple: triple[0] not in {MS.And, MS.Or}, repeated_combinators
    )
    repeated_combinators = {subj: obj for subj, _, obj in repeated_combinators}

    intro_terms = list(restriction_rdf_graph.subjects(MS.belongsTo, MS.IntroTerm))

    restriction_rdf_graph.remove((None, RDF.type, None))  # TODO: exempt if intro term
    restriction_rdf_graph.remove((None, SKOS.exactMatch, None))
    restriction_rdf_graph.remove((None, MS.belongsTo, None))
    graph_triples = (
        (subj, obj, {"label": pred}) for subj, pred, obj in restriction_rdf_graph
    )
    graph_triples = filterfalse(
        lambda triple: isinstance(triple[0], BNode), graph_triples
    )
    graph.add_edges_from(list(graph_triples))
    graph.edges(data=True)
    restriction_rdf_graph.serialize("intermediate.ttl", format="turtle")

    # TODO: add mapping to exact matches in namespace class generator script
    ms_ttl_term_mapping = {
        MS.equivalentTo: OWL.equivalentClass,
        MS.some: OWL.someValuesFrom,
        MS.of: OWL.onClass,
        MS.max: OWL.maxQualifiedCardinality,
        MS.min: OWL.minQualifiedCardinality,
        MS.only: OWL.allValuesFrom,
    }

    # initiate chain
    # NOTE: the chains only apply to property restrictions!
    # TODO: add support for just datatype. Datatype facets for property restrictions are supported.
    expanded_axiom_rdf_graph = rdflib.Graph()
    for prefix, namespace_uri in restriction_rdf_graph.namespaces():
        expanded_axiom_rdf_graph.bind(prefix, namespace_uri)

    pivot_node_types = dict()
    pivot_triples = restriction_rdf_graph.triples_choices(
        (None, MS.forWhich, list(repeated_combinators.keys()))
    )
    pivot_node_predicates = map_reduce(
        pivot_triples,
        keyfunc=lambda triple: triple[2],
        valuefunc=lambda triple: restriction_rdf_graph.predicates(triple[2], None),
    )
    class_rest_preds = {MS.equivalentTo, RDF.type, RDFS.subClassOf}
    prop_rest_preds = {MS.max, MS.only, MS.that, MS.exactly, MS.value}
    # TODO: check that the restriction graph has no floating nodes
    # TODO: check if the predicates are all the same type as a preprocessing step
    # TODO: handle unknown case
    for key, values in pivot_node_predicates.items():
        type_det_value = next(flatten(values))
        pred_type = None
        if type_det_value in class_rest_preds:
            pred_type = "class"
        elif type_det_value in prop_rest_preds:
            pred_type = "prop"
        pivot_node_types[key] = pred_type
    compressed_graph = graph.copy()
    compressed_nodes = defaultdict(list)
    visited_combinator = set()
    for intro_term in intro_terms:
        current_node = None
        for subj, obj, label in nx.dfs_labeled_edges(graph, intro_term):
            if subj == obj:
                continue
            pred = graph[subj][obj].get("label", None) if subj != obj else None
            print(subj, obj, pred, label)
            if label == 'forward':
                if pred == MS.forWhich:
                    continue
                if subj in repeated_combinators and subj not in visited_combinator:
                    comb_subj = next(graph.predecessors(obj))
                    compressed_node = BNode()
                    compressed_node_attrs = {"subject": comb_subj, "combinator": repeated_combinators[subj], "combinator_type": pivot_node_types[subj]}
                    compressed_graph.add_node(compressed_node, **compressed_node_attrs)
                    compressed_nodes[compressed_node] = [subj]
                    visited_combinator.add(subj)
                if obj in repeated_combinators:
                    current_node = None
                else:
                    if current_node is None:
                        current_node = BNode()
                    compressed_nodes[current_node].append((pred, obj))
            else:
                current_node = None

    pprint(compressed_nodes)
    # for intro_term in intro_terms:
    #     restriction = BNode()
    #     for subj, obj in nx.edge_dfs(graph, intro_term):
    #         label = graph[subj][obj].get("label", None)
    #         if subj == intro_term:
    #             ttl_pred = ms_ttl_term_mapping[label]
    #             expanded_axiom_rdf_graph.add((intro_term, ttl_pred, restriction))
    #             print(list(unique_everseen(chain(types, classes))))
    #             expanded_axiom_rdf_graph.add((restriction, RDF.type, OWL.Restriction))
    #             expanded_axiom_rdf_graph.add((restriction, OWL.onProperty, obj))
    #         elif label in ms_ttl_term_mapping: # TODO: support turtle value counterpart as well
    #             ttl_pred = ms_ttl_term_mapping[label]
    #             if isinstance(obj, BNode):
    #                 relevant_nodes = restriction_rdf_graph.transitive_objects(obj, None)
    #                 relevant_nodes = filter(lambda node: isinstance(node, BNode), relevant_nodes)
    #                 collection_triples = flatten(map(lambda node: restriction_rdf_graph.triples((node, None, None)), relevant_nodes))
    #                 for collection_triple in collection_triples:
    #                     expanded_axiom_rdf_graph.add(collection_triple)
    #             elif isinstance(obj, Literal) and isinstance(obj.value, str) and (match := re.search(r'(.*)\[(.*)\]', obj.value)):
    #                 facet = match.group(2)
    #                 value = match.group(1).strip()
    #                 obj = Literal(value)
    #             expanded_axiom_rdf_graph.add((restriction, ttl_pred, obj))
    expanded_axiom_rdf_graph.serialize("axiom_intermediate.ttl", format="turtle")

    # remove original chain triples

    return restriction_rdf_graph
