import re
from collections import defaultdict
from functools import partial
from itertools import chain

import networkx as nx
import pandas as pd
from more_itertools.recipes import flatten
from networkx import DiGraph
from rdflib import RDF, Graph, BNode, OWL, URIRef, Literal, RDFS, XSD
from rdflib.collection import Collection

from cemento.axioms.constants import get_ms_turtle_mapping, symbol_mapping
from cemento.axioms.modules import MS
from cemento.rdf.transforms import (
    get_uuid,
)
from cemento.term_matching.constants import FuzzAlgorithm
from cemento.term_matching.transforms import substitute_term
from cemento.utils.constants import invert_tuple
from cemento.utils.utils import fst, get_subgraphs


def parse_item(
    item, collection_headers, term_substitution, facet_substitution, ms_turtle_mapping
) -> URIRef:
    parsed_item = (
        facet_substitution.get(item, None)
        or collection_headers.get(item, None)
        or term_substitution.get(item)
    )
    parsed_item = ms_turtle_mapping.get(parsed_item, parsed_item)
    return parsed_item


def parse_chain_tuple(
    input_tuple,
    collection_headers,
    term_substitution,
    facet_substitution,
    ms_turtle_mapping,
):
    parse_tuple_item = partial(
        parse_item,
        collection_headers=collection_headers,
        term_substitution=term_substitution,
        facet_substitution=facet_substitution,
        ms_turtle_mapping=ms_turtle_mapping,
    )
    return tuple(map(parse_tuple_item, input_tuple))


def infer_number_str_type(number_str: str) -> URIRef:
    try:
        number_str_sample = pd.to_numeric(number_str, errors="raise")
    except ValueError as e:
        raise ValueError("The input facet number is not a number.") from e
    number_str_class = str(type(number_str_sample))
    value_type = XSD.decimal if "float" in number_str_class else XSD.integer
    return value_type


def retrieve_facet_nodes(
    facet_graph, term_substitution, faceted_terms
) -> tuple[Graph, dict[str, BNode]]:
    facet_nodes = dict()
    for key, facet in faceted_terms.items():
        facet = re.match(r".*\[(.*)\]", facet).group(1)
        facet_pairs = list(
            map(lambda item: item.strip().split(" "), facet.strip().split(","))
        )
        if any(len(pair) != 2 for pair in facet_pairs):
            raise ValueError(
                "Please make sure to separate symbols in facets with spaces and commas"
            )
        facet_term_search_pool = set(invert_tuple(symbol_mapping.items()))

        restriction_node = BNode()
        collection_head = BNode()
        facet_nodes[key] = restriction_node
        facet_graph.add((restriction_node, RDF.type, RDFS.Datatype))
        facet_graph.add((restriction_node, OWL.onDatatype, term_substitution[key]))
        facet_graph.add((restriction_node, OWL.withRestrictions, collection_head))
        facet_collection = Collection(facet_graph, collection_head)
        for term, value in facet_pairs:
            facet_key = substitute_term(
                term, facet_term_search_pool, search_algorithm=FuzzAlgorithm.SimpleRatio
            )
            if facet_key is None:
                raise ValueError(
                    f"The facet key must be one of: {symbol_mapping.keys()}"
                )
            item_bnode = BNode()
            facet_graph.add(
                (
                    item_bnode,
                    facet_key,
                    Literal(value, datatype=infer_number_str_type(value)),
                )
            )
            facet_collection.append(item_bnode)
        return facet_graph, facet_nodes


def expand_tree(
    term_graph, restriction_nodes, pivot_nodes
) -> tuple[DiGraph, dict[str, URIRef]]:
    ## expand the tree to include relevant pivots
    ### expand for non-pivot nodes that have more than one descendant
    triples_to_add = []
    triples_to_remove = []
    node_labels = dict()
    for restriction_node in restriction_nodes:
        for node in nx.bfs_tree(term_graph, source=restriction_node):
            descendants = list(term_graph.successors(node))
            if len(descendants) > 1 and node not in pivot_nodes:
                new_pivot_node = get_uuid()
                node_labels[new_pivot_node] = MS.And  # default to intersection
                triples_to_add.append((node, new_pivot_node))
                triples_to_add.extend(
                    [
                        (new_pivot_node, child, term_graph[node][child])
                        for child in descendants
                    ]
                )
                triples_to_remove.extend([(node, child) for child in descendants])

    term_graph.remove_edges_from(triples_to_remove)
    term_graph.add_edges_from(triples_to_add)

    pivot_nodes.update(set(node_labels.keys()))

    ### add a pivot node for remaining trees with no pivots
    triples_to_add, triples_to_remove = [], []
    for restriction_node in restriction_nodes:
        next_successor = next(term_graph.successors(restriction_node), None)
        if next_successor not in pivot_nodes:
            new_pivot_node = get_uuid()
            node_labels[new_pivot_node] = MS.Single
            triples_to_add.append((restriction_node, new_pivot_node))
            triples_to_add.append(
                (
                    new_pivot_node,
                    next_successor,
                    term_graph[restriction_node][next_successor],
                )
            )
            triples_to_remove.append((restriction_node, next_successor))

    term_graph.remove_edges_from(triples_to_remove)
    term_graph.add_edges_from(triples_to_add)

    return term_graph, node_labels


def extract_axiom_graph(
    term_graph,
    term_substitution,
    restriction_nodes,
    collection_headers,
    intro_restriction_triples,
    faceted_terms,
) -> Graph:
    # pre-retrieve all relevant terms
    pivot_terms = {MS.And, MS.Or, MS.Single}
    pivot_nodes = filter(lambda item: item[1] in pivot_terms, term_substitution.items())
    pivot_nodes = set(map(fst, pivot_nodes))
    ms_turtle_mapping = get_ms_turtle_mapping()

    term_graph, node_labels = expand_tree(term_graph, restriction_nodes, pivot_nodes)
    ### update variables for expanded graph
    pivot_nodes.update(set(node_labels.keys()))
    term_substitution.update(node_labels)
    head_nodes = flatten(
        map(lambda node: term_graph.successors(node), restriction_nodes)
    )
    head_nodes = set(head_nodes)

    ## traverse the axiom subgraphs and make connections
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
    facet_graph, facet_nodes = Graph(), dict()

    if faceted_terms and len(faceted_terms) > 0:
        facet_graph, facet_nodes = retrieve_facet_nodes(
            facet_graph, term_substitution, faceted_terms
        )
        axiom_graph += facet_graph

    parse_axiom_item = partial(
        parse_item,
        collection_headers=collection_headers,
        term_substitution=term_substitution,
        facet_substitution=facet_nodes,
        ms_turtle_mapping=ms_turtle_mapping,
    )
    parse_axiom_tuple = partial(
        parse_chain_tuple,
        collection_headers=collection_headers,
        term_substitution=term_substitution,
        facet_substitution=facet_nodes,
        ms_turtle_mapping=ms_turtle_mapping,
    )

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
                    pred, obj = parse_axiom_tuple(apply_tuple)
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
                    pred, obj = parse_axiom_tuple((pred, obj))
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
                    axiom_graph.add((header, parsed_label, collection_bnode))
                    axiom_graph.add((header, RDF.type, OWL.Class))
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

    ## translate pattern-based Manchester constructs
    ### replace for property chains
    property_chain_properties = [OWL.propertyChainAxiom, MS.subPropertyChain]
    replace_triples = list(
        axiom_graph.triples_choices((None, property_chain_properties, None))
    )
    replace_tuples = set(map(lambda item: (item[0], item[2]), replace_triples))
    for subj, obj in replace_tuples:
        collection_node = BNode()
        collection_members = [obj] + list(
            axiom_graph.transitive_objects(subject=obj, predicate=MS.o)
        )
        Collection(axiom_graph, collection_node, collection_members)
        axiom_graph.add((subj, OWL.propertyChainAxiom, collection_node))
    ### remove old triples
    o_triples = axiom_graph.triples((None, MS.o, None))
    for triple in chain(replace_triples, o_triples):
        axiom_graph.remove(triple)
    return axiom_graph
