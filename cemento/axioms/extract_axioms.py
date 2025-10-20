from collections import defaultdict
from functools import partial

import networkx as nx
from more_itertools.recipes import flatten
from networkx import DiGraph
from rdflib import RDF, Graph, BNode, OWL, URIRef
from rdflib.collection import Collection

from cemento.axioms.constants import get_ms_turtle_mapping
from cemento.axioms.modules import MS
from cemento.rdf.transforms import (
    get_uuid,
)
from cemento.utils.utils import fst, get_subgraphs


def parse_item(
    item, collection_headers, term_substitution, ms_turtle_mapping
) -> URIRef:
    parsed_item = collection_headers.get(item, None) or term_substitution.get(item)
    parsed_item = ms_turtle_mapping.get(parsed_item, parsed_item)
    return parsed_item


def parse_chain_tuple(
    collection_headers, term_substitution, ms_turtle_mapping, input_tuple
):
    parse_tuple_item = partial(
        parse_item,
        collection_headers=collection_headers,
        term_substitution=term_substitution,
        ms_turtle_mapping=ms_turtle_mapping,
    )
    return tuple(map(parse_tuple_item, input_tuple))


def extract_axiom_graph(
    rdf_graph,
    term_graph,
    term_substitution,
    restriction_nodes,
    collection_headers,
    intro_restriction_triples,
) -> Graph:
    pivot_terms = {MS.And, MS.Or, MS.Single}
    pivot_nodes = filter(lambda item: item[1] in pivot_terms, term_substitution.items())
    pivot_nodes = set(map(fst, pivot_nodes))
    head_nodes = flatten(
        map(lambda node: term_graph.successors(node), restriction_nodes)
    )
    ms_turtle_mapping = get_ms_turtle_mapping()
    parse_axiom_item = partial(
        parse_item,
        collection_headers=collection_headers,
        term_substitution=term_substitution,
        ms_turtle_mapping=ms_turtle_mapping,
    )
    chain_containers = defaultdict(list)
    pivot_chain_mapping = defaultdict(list)
    compressed_graph = DiGraph()
    for head_node in head_nodes:
        current_node = None
        for subj, obj in nx.dfs_edges(term_graph, source=head_node):
            pred = term_graph[subj][obj].get("label", None)
            if subj in pivot_nodes:
                current_node = get_uuid()
                pivot_chain_mapping[subj].append(current_node)
                compressed_graph.add_edge(subj, current_node)
            if obj in pivot_nodes:
                compressed_graph.add_edge(current_node, obj)
                continue
            chain_containers[current_node].append((pred, obj))

    # FIXME: find a way to pass multiple bnode headers and process them
    # FIXME: compressed graph edges between a node and a pivot not being added correctly
    axiom_graph = Graph()
    compressed_subtrees = get_subgraphs(compressed_graph)
    axiom_combination_bnodes = dict()
    axiom_header = dict()
    for tree in compressed_subtrees:
        for node in nx.dfs_postorder_nodes(tree):
            if node not in chain_containers and node not in pivot_nodes:
                raise ValueError(
                    f"the element with id {node} must either be a chain container or a pivot node."
                )
            header = BNode()
            axiom_header[node] = header
            if node in chain_containers:
                successor_pivots = list(
                    filter(
                        lambda item: item in pivot_chain_mapping, tree.successors(node)
                    )
                )
                if len(successor_pivots) > 0:
                    apply_tuple = chain_containers[node].pop()
                    pred, obj = parse_chain_tuple(
                        collection_headers,
                        term_substitution,
                        ms_turtle_mapping,
                        apply_tuple,
                    )
                    successor_pivot_bnodes = flatten(
                        map(
                            lambda node: axiom_combination_bnodes[node],
                            successor_pivots,
                        )
                    )
                    for bnode in successor_pivot_bnodes:
                        axiom_graph.add((bnode, pred, obj))

                    axiom_header[node] = axiom_header[next(iter(successor_pivots))]
                for pred, obj in chain_containers[node]:
                    axiom_graph.add((header, RDF.type, OWL.Restriction))
                    pred, obj = parse_chain_tuple(
                        collection_headers,
                        term_substitution,
                        ms_turtle_mapping,
                        (pred, obj),
                    )
                    axiom_graph.add((header, pred, obj))
            else:
                pivot_node_children = tree.successors(node)
                child_bnodes = list(
                    map(lambda item: axiom_header[item], pivot_node_children)
                )
                axiom_combination_bnodes[node] = child_bnodes
                if len(child_bnodes) > 1:
                    collection_bnode = BNode()
                    Collection(axiom_graph, collection_bnode, child_bnodes)
                    parsed_label = parse_axiom_item(node)
                    rdf_graph.add((header, parsed_label, collection_bnode))
                else:
                    axiom_header[node] = next(iter(child_bnodes), header)

    outgoing_restriction_triples = filter(
        lambda triple: triple[0] in restriction_nodes, term_graph.edges
    )
    outgoing_tuple_mapping = {subj: obj for subj, obj in outgoing_restriction_triples}
    for subj, obj, data in intro_restriction_triples:
        obj = axiom_header[outgoing_tuple_mapping[obj]]
        subj = parse_axiom_item(subj)
        pred = parse_axiom_item(data["label"])
        axiom_graph.add((subj, pred, obj))
    return axiom_graph
