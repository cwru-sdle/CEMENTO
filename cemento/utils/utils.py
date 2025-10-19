import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import reduce

import networkx as nx
from networkx import DiGraph

from cemento.utils.constants import NullTermError


def fst(x: tuple[any, any]) -> any:
    return x[0]


def snd(x: tuple[any, any]) -> any:
    return x[1]


def trd(x: tuple[any, any, any]) -> any:
    return x[2]


def remove_term_names(term: str) -> str:
    match = re.search(r"^([^(]*)", term)
    return match.group(1).strip() if match else term


def aggregate_defaultdict(
    acc: defaultdict[list], item: tuple[any, any]
) -> defaultdict[list]:
    key, value = item
    # TODO: implement immutable copy here
    acc[key].append(value)
    return acc


def chain_filter(iterable: Iterable, *filters: Callable[[any], any]):
    return reduce(
        lambda iterable, filter_func: filter(filter_func, iterable),
        filters,
        iterable,
    )


def get_abbrev_term(
    term: str, is_predicate=False, default_prefix="mds"
) -> tuple[str, str]:
    prefix = default_prefix
    abbrev_term = term
    strict_camel_case = False

    if term is None or not term:
        raise NullTermError(
            "There is a null term. Maybe you forgot to label something?"
        )

    term = remove_term_names(term)
    if ":" in term:
        prefix, abbrev_term = term.split(":")

    if is_predicate:
        abbrev_term = abbrev_term.replace("_", " ")
        strict_camel_case = True

    # if the term is a class, use upper camel case / Pascal case
    abbrev_term = "".join(
        [
            f"{word[0].upper()}{word[1:] if len(word) > 1 else ''}"
            for word in abbrev_term.split()
        ]
    )

    if strict_camel_case:
        abbrev_term = (
            f"{abbrev_term[0].lower()}{abbrev_term[1:] if len(abbrev_term) > 1 else ''}"
        )

    return prefix, abbrev_term


# TODO: assign graph utils to their own module
def filter_graph(
    graph: DiGraph, data_filter: Callable[[dict[str, any]], bool]
) -> DiGraph:
    filtered_graph = graph.copy()
    filtered_graph.remove_edges_from(
        [
            (subj, obj)
            for subj, obj, data in graph.edges(data=True)
            if (not data_filter(data) if data_filter else False)
        ]
    )
    return filtered_graph


# TODO: move to utils
def get_subgraphs(graph: DiGraph) -> list[DiGraph]:
    subgraphs = nx.weakly_connected_components(graph)
    return [graph.subgraph(subgraph_nodes).copy() for subgraph_nodes in subgraphs]


# TODO: move to utils
def get_graph_root_nodes(graph: DiGraph) -> list[any]:
    return [
        node
        for node in graph.nodes
        if graph.in_degree(node) == 0 or len(graph.nodes) == 1
    ]
