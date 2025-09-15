from pprint import pprint
import re
from collections.abc import Callable, Iterable
from functools import reduce
from itertools import chain, groupby
from uuid import uuid4

import networkx as nx
from networkx import DiGraph
from rdflib import OWL, RDF, RDFS, SKOS, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import split_uri

from cemento.axioms.modules import MDS
from cemento.rdf.preprocessing import (
    clean_literal_string,
    format_literal,
    remove_suppression_key,
)
from cemento.term_matching.constants import RANK_PROPS
from cemento.term_matching.transforms import substitute_term_multikey
from cemento.utils.constants import valid_collection_types
from cemento.axioms.modules import MS
from cemento.utils.utils import (
    enforce_camel_case,
    filter_graph,
    fst,
    snd,
)


def construct_term_uri(
    prefix: str,
    abbrev_term: str,
    prefixes: dict[str, URIRef | Namespace],
) -> URIRef:
    ns_uri = prefixes[prefix]
    abbrev_term = remove_suppression_key(abbrev_term)
    return URIRef(f"{ns_uri}{abbrev_term}")


def get_xsd_terms() -> dict[str, URIRef]:
    terms = list(filter(lambda term: isinstance(term, URIRef), dir(XSD)))
    abbrev_terms = map(lambda term: f"xsd:{snd(split_uri(term))}", terms)
    return {
        abbrev_term: term
        for (abbrev_term, term) in zip(abbrev_terms, terms, strict=True)
    }


def construct_literal(term: str, lang="en", datatype=None) -> Literal:
    return Literal(clean_literal_string(term), lang=lang, datatype=datatype)


def get_literal_lang_annotation(literal_term: str, default=None) -> str:
    return res[0] if (res := re.findall(r"@(\w+)", literal_term)) else default


def get_literal_data_type(
    literal_term: str,
    search_terms: dict[str, URIRef],
    score_cutoff=90,
) -> URIRef | None:
    search_key = res[0] if (res := re.findall(r"\^\^(\w+:\w+)", literal_term)) else None
    if search_key:
        datatype = substitute_term_multikey(
            [search_key], search_terms, score_cutoff=score_cutoff
        )
        return datatype
    return None


def get_class_terms(graph: DiGraph) -> set[URIRef]:
    subclass_terms = {
        term
        for subj, obj, data in graph.edges(data=True)
        if data["label"] == RDFS.subClassOf
        for term in (subj, obj)
    }
    type_terms = {
        obj for subj, obj, data in graph.edges(data=True) if data["label"] == RDF.type
    }
    return subclass_terms | type_terms


def get_term_value(subj: URIRef, pred: URIRef, ref_rdf_graph: Graph):
    return ref_rdf_graph.value(subj, pred)


def bind_prefixes(rdf_graph: Graph, prefixes: dict[str, URIRef | Namespace]) -> Graph:
    for prefix, ns in prefixes.items():
        rdf_graph.bind(prefix, ns)
    return rdf_graph


def add_rdf_triples(
    rdf_graph: Graph,
    triples: Iterable[tuple[URIRef | Literal, URIRef, URIRef | Literal]],
) -> Graph:
    # TODO: set to strictly immutable rdf_graph
    return reduce(lambda graph, triple: graph.add(triple), triples, rdf_graph)


def add_labels(rdf_graph: Graph, term: URIRef, labels: list[str]) -> Graph:
    # assume the first element of the labels is the actual label, others are alt-names
    # TODO: set to strictly immutable rdf_graph
    if labels:
        rdf_graph = rdf_graph.add((term, RDFS.label, Literal(labels[0])))
        if len(labels) > 1:
            rdf_graph = add_rdf_triples(
                rdf_graph,
                ((term, SKOS.altLabel, Literal(label)) for label in labels[1:]),
            )
    return rdf_graph


def get_term_domain(term: URIRef, graph: DiGraph) -> Iterable[URIRef]:
    return (subj for subj, _, data in graph.edges(data=True) if data["label"] == term)


def get_term_range(term: URIRef, graph: DiGraph) -> Iterable[URIRef]:
    return (obj for _, obj, data in graph.edges(data=True) if data["label"] == term)


def get_domains_ranges(
    predicate: Iterable[URIRef], graph: DiGraph
) -> tuple[URIRef, set[URIRef], set[URIRef]]:
    return (
        predicate,
        set(get_term_domain(predicate, graph)),
        set(get_term_range(predicate, graph)),
    )


def get_term_collection_triples(
    rdf_graph: Graph,
    head_term: URIRef,
    member_terms: Iterable[URIRef],
    member_rel: URIRef,
    term_collection_rel: URIRef,
) -> list[tuple[URIRef, URIRef, URIRef]]:
    triples = []
    collection_node = BNode()
    Collection(rdf_graph, collection_node, member_terms)
    # create class that points to the collection
    collection_class = BNode()
    triples.append((collection_class, RDF.type, OWL.Class))
    # connect them all together
    triples.append((collection_class, member_rel, collection_node))
    triples.append((head_term, term_collection_rel, collection_class))
    return triples


def add_domains_ranges(
    term_domains_ranges: [URIRef, Iterable[URIRef], Iterable[URIRef]],
    rdf_graph: DiGraph,
) -> Graph:
    predicate_term, domains, ranges = term_domains_ranges
    # TODO: assume union for now but fix later
    domain_collection_triples = [(predicate_term, RDFS.domain, next(iter(domains)))]
    if len(domains) > 1:
        domain_collection_triples = get_term_collection_triples(
            rdf_graph, predicate_term, domains, OWL.unionOf, RDFS.domain
        )

    range_collection_triples = [(predicate_term, RDFS.range, next(iter(ranges)))]
    if len(ranges) > 1:
        range_collection_triples = get_term_collection_triples(
            rdf_graph, predicate_term, ranges, OWL.unionOf, RDFS.range
        )
    return add_rdf_triples(
        rdf_graph, domain_collection_triples + range_collection_triples
    )


def remove_generic_property(
    rdf_graph: Graph, default_property: URIRef = RDF.Property
) -> Graph:
    # TODO: implement immutable copy of rdf_graph here
    generic_pred_subjects = list(rdf_graph.subjects(None, default_property))
    subject_property_type_defs = rdf_graph.triples_choices(
        (generic_pred_subjects, RDF.type, None)
    )
    sorted_property_defs = sorted(subject_property_type_defs, key=lambda x: fst(x))
    subject_property_defs_ct = (
        (subj, len(list(objs)))
        for subj, objs in groupby(sorted_property_defs, key=lambda x: x[0])
    )
    rdf_graph = reduce(
        lambda rdf_graph, subject_info: (
            rdf_graph.remove((fst(subject_info), RDF.type, default_property))
            if snd(subject_info) > 1
            else rdf_graph
        ),
        subject_property_defs_ct,
        rdf_graph,
    )
    return rdf_graph


def get_graph_relabel_mapping(
    terms: URIRef,
    all_classes: set[URIRef],
    all_instances: set[URIRef],
    aliases: dict[URIRef, Literal],
    inv_prefix: dict[str, str],
) -> dict[URIRef, str]:
    rename_mapping = dict()
    for term in terms:
        ns, abbrev_term = split_uri(term)
        prefix = inv_prefix[str(ns)]
        new_name = f"{prefix}:{abbrev_term}"
        if term in aliases and aliases[term]:
            if term in all_classes or term in all_instances:
                new_name += f" ({','.join(aliases[term])})"
            else:
                new_name = f"{prefix}:{aliases[term][0]}"
        rename_mapping[term] = new_name
    return rename_mapping


def get_literal_prefix(
    literal: Literal, inv_prefixes: dict[URIRef | Namespace, str]
) -> str:
    if hasattr(literal, "datatype") and literal.datatype:
        ns, _ = split_uri(literal.datatype)
        prefix = inv_prefixes.get(ns, None)
        return prefix
    elif literal.value:
        # default to xsd, since non-annotated terms are strings
        return "xsd"
    return None


def get_literal_format_mapping(
    graph: DiGraph, inv_prefixes: dict[URIRef | Namespace, str]
) -> dict[Literal, str]:
    return {
        literal: format_literal(literal, get_literal_prefix(literal, inv_prefixes))
        for literal in filter(lambda term: isinstance(term, Literal), graph.nodes)
    }


def add_triples_to_digraph(
    subj: URIRef | Literal,
    pred: URIRef,
    obj: URIRef | Literal,
    graph: DiGraph,
) -> DiGraph:
    new_graph = graph.copy()
    new_graph.add_edge(subj, obj, label=pred)
    return new_graph


def assign_edge_attr(
    graph: DiGraph, edges: tuple[any, any], new_attrs: dict[str, any]
) -> DiGraph:
    new_graph = graph.copy()
    edge_attr_dict = {edge: new_attrs for edge in edges}
    nx.set_edge_attributes(new_graph, edge_attr_dict)
    return new_graph


def assign_edge_binary_attr(
    graph: Graph, filter_func: Callable[[dict[str, any]], bool], attr: str
):
    new_graph = graph.copy()
    positive_graph = filter_graph(
        graph,
        filter_func,
    )
    negative_edges = graph.edges - positive_graph.edges
    new_graph = assign_edge_attr(new_graph, positive_graph.edges, {attr: True})
    new_graph = assign_edge_attr(new_graph, negative_edges, {attr: False})
    return new_graph


def assign_rank_status(graph: DiGraph, rank_terms: set[URIRef] = RANK_PROPS):
    return assign_edge_binary_attr(
        graph, lambda data: data["label"] in rank_terms, "is_rank"
    )


def assign_pred_status(graph: DiGraph) -> DiGraph:
    # all edges are predicates
    return assign_edge_binary_attr(graph, lambda data: True, "is_predicate")


def assign_strat_status(
    graph: DiGraph, strat_terms: set[URIRef] = RANK_PROPS
) -> DiGraph:
    return assign_edge_binary_attr(
        graph, lambda data: data["label"] in strat_terms, "is_strat"
    )


def assign_literal_status(graph: DiGraph, all_literals: set[Literal]) -> DiGraph:
    new_graph = graph.copy()
    node_values = {node: {"is_literal": node in all_literals} for node in graph.nodes}
    nx.set_node_attributes(new_graph, node_values)
    return new_graph


def rename_edges(graph: DiGraph, rename_mapping: dict[URIRef, str]) -> DiGraph:
    edge_rename_mapping = dict()
    graph = graph.copy()
    for subj, obj, data in graph.edges(data=True):
        pred = data["label"]
        new_edge_label = rename_mapping[pred]
        data.update({"label": new_edge_label})
        edge_rename_mapping[(subj, obj)] = data
    nx.set_edge_attributes(graph, edge_rename_mapping)
    return graph


def get_literal_values_with_id(
    literal_terms: list[Literal],
) -> Iterable[tuple[Literal, Literal]]:
    # TODO: add a hashed version of the tag literal_id- to prevent conflict if people put this string
    unique_ids = (f"literal_id-{get_uuid()}" for _ in range(len(literal_terms)))
    return (
        (
            literal,
            Literal(
                f"{unique_id}:{literal.value}",
                lang=literal.language if hasattr(literal, "language") else None,
                datatype=literal.datatype if hasattr(literal, "datatype") else None,
            ),
        )
        for (unique_id, literal) in zip(unique_ids, literal_terms, strict=True)
    )


def assign_literal_ids(
    rdf_graph: Graph, literal_replacements: Iterable[tuple[Literal, Literal]]
) -> Graph:
    literal_map = dict(literal_replacements)  # old_literal -> new_literal

    for old_literal, new_literal in literal_map.items():
        for subj, pred, obj in list(rdf_graph.triples((None, None, old_literal))):
            rdf_graph.remove((subj, pred, obj))
            rdf_graph.add((subj, pred, new_literal))

    return rdf_graph


def get_uuid():
    return str(uuid4()).split("-")[-1]


def get_collection_nodes(graph: DiGraph) -> dict[str, str]:
    collections_in_graph = filter(lambda x: x in graph.nodes, valid_collection_types)
    collection_nodes = {
        node: collection_type
        for collection_type in collections_in_graph
        for node in graph.neighbors(collection_type)
    }
    return collection_nodes


def get_collection_subgraph(collection_nodes: set[str], graph: DiGraph):
    collection_node_refs = collection_nodes | valid_collection_types
    collection_members = {
        member for node in collection_nodes for member in graph.neighbors(node)
    }
    return graph.subgraph(collection_node_refs | collection_members).copy()


def get_collection_in_edges(
    collection_nodes: set[str], graph: DiGraph
) -> list[tuple[str, str, dict[str, str | bool]]]:
    return [
        (subj, obj, attr)
        for collection_id in collection_nodes
        for (subj, obj, attr) in graph.in_edges(collection_id, data=True)
        if subj not in valid_collection_types
        and ("label" not in attr or attr["label"] != "mds:hasCollectionMember") # TODO: convert string comparisons with namespace-class based symbol
    ]


def get_collection_members(
    collection_subgraph: DiGraph,
    collection_id: str,
    term_mapping: dict[str, URIRef | Literal],
) -> Iterable[str]:
    return (
        term_mapping[member] if member in term_mapping else member
        for _, member in collection_subgraph.out_edges(collection_id)
    )


def get_collection_nodes_iter(
    collection_subgraph: DiGraph, collection_nodes: set[str]
) -> Iterable[str]:
    collection_subgraph_postorder_nodes = nx.dfs_postorder_nodes(collection_subgraph)
    return filter(lambda x: x in collection_nodes, collection_subgraph_postorder_nodes)


# TODO: decouple this function even further, ideally with one for each current output
def get_collection_triples_and_targets(
    collection_nodes: dict[str, str],
    collection_subgraph: DiGraph,
    rdf_graph: Graph,
    term_mapping: dict[str, URIRef | Literal],
):
    container_refs = dict()
    collection_triples = []
    collection_type_map = {
        "owl:unionOf": OWL.unionOf,
        "owl:intersectionOf": OWL.intersectionOf,
        "owl:complementOf": OWL.complementOf,
    }
    collection_nodes_iter = get_collection_nodes_iter(
        collection_subgraph, collection_nodes
    )
    for collection_id in collection_nodes_iter:
        # swap out string id with the corresponding constructed term
        # return the same member if not in the constructed term dict, especially for ids
        members = get_collection_members(
            collection_subgraph, collection_id, term_mapping
        )
        # swap out collection members for their constructed BNode
        members = [
            container_refs[member] if member in container_refs else member
            for member in members
        ]
        collection_type_str = collection_nodes[collection_id]
        collection_type = (
            collection_type_map[collection_type_str]
            if collection_type_str in collection_type_map
            else None
        )
        # create the collection and refer to the node
        if collection_type:
            collection_node = BNode()
            Collection(rdf_graph, collection_node, members)
            collection_class = BNode()
            collection_triples.append(
                (collection_class, collection_type, collection_node)
            )
            container_refs[collection_id] = collection_class
        else:
            # assign the members to directly map as a flat collection
            container_refs[collection_id] = members

    return collection_triples, container_refs


def add_collection_links_to_graph(
    collection_in_edges: list[tuple[str, str, dict[str, str | bool]]],
    collection_targets: dict[str, list[URIRef | Literal] | BNode],
    graph: DiGraph,
):
    graph = graph.copy()
    for subj, obj, data in collection_in_edges:
        # if the reference is a list of more than one element, just use flat mapping
        if isinstance(collection_targets[obj], list):
            members = collection_targets[obj]
            for member in members:
                graph.add_edge(subj, member, label=data["label"])
        else:
            graph.add_edge(
                subj,
                collection_targets[obj],
                label=data["label"],
            )
    return graph



def enforce_camel_case_in_rdf_graph(
    rdf_graph: Graph, terms_to_replace: Iterable[URIRef]
):
    # TODO: make this more functional, implement immutable rdf_graph copy
    for term in terms_to_replace:
        all_triples_with_term = set(
            chain(
                rdf_graph.triples((term, None, None)),
                rdf_graph.triples((None, term, None)),
                rdf_graph.triples((None, None, term)),
            )
        )
        for subj, pred, obj in all_triples_with_term:
            ns, abbrev_term = split_uri(term)
            new_term = URIRef(f"{ns}{enforce_camel_case(abbrev_term)}")
            replacement = {term: new_term}
            rdf_graph.remove((subj, pred, obj))
            rdf_graph.add(
                (
                    replacement.get(subj, subj),
                    replacement.get(pred, pred),
                    replacement.get(obj, obj),
                )
            )
    return rdf_graph
