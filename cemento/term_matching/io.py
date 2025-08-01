import json
import os
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from rdflib import RDFS, SKOS, Graph, Namespace, URIRef
from rdflib.namespace import split_uri


def get_ttl_file_iter(folder_path: str | Path) -> Iterable[Graph]:
    return (
        get_ttl_graph(file_path)
        for file in os.scandir(folder_path)
        if (file_path := Path(file.path)).suffix == ".ttl"
    )


def get_ttl_graph(file_path: str | Path) -> Graph | None:
    with read_ttl(file_path) as graph:
        return graph


@contextmanager
def read_ttl(file_path: str | Path) -> Graph:
    rdf_graph = Graph()
    try:
        rdf_graph.parse(file_path, format="turtle")
        yield rdf_graph
    finally:
        rdf_graph.close()


def read_prefixes_from_json(file_path: str) -> dict[str, URIRef]:
    with open(file_path, "r") as f:
        prefixes = json.load(f)
        return prefixes


def get_search_terms_from_defaults(
    default_namespace_prefixes: dict[str, Namespace],
) -> dict[str, URIRef]:
    search_terms = dict()
    for prefix, ns in default_namespace_prefixes.items():
        for term in dir(ns):
            if isinstance(term, URIRef):
                _, name = split_uri(term)
                search_terms[f"{prefix}:{name}"] = term
    return search_terms


def read_prefixes_from_graph(rdf_graph: Graph) -> dict[str, str]:
    return {prefix: str(ns) for prefix, ns in rdf_graph.namespaces()}


def get_search_terms_from_graph(
    rdf_graph: Graph, inv_prefixes: dict[str, str]
) -> dict[str, URIRef]:
    search_terms = dict()
    all_terms = set()
    for subj, pred, obj in rdf_graph:
        all_terms.update([subj, pred, obj])

        # TODO: take comparison set from constnats
        if pred == RDFS.label or pred == SKOS.altLabel:
            ns, _ = split_uri(subj)
            prefix = inv_prefixes[ns]
            search_terms[f"{prefix}:{str(obj)}"] = subj

    for term in all_terms:
        if isinstance(term, URIRef):
            is_literal = False
            try:
                ns, abbrev_term = split_uri(term)
            except ValueError:
                is_literal = not is_literal

            if not is_literal and str(ns) in inv_prefixes:
                prefix = inv_prefixes[str(ns)]
                search_terms[f"{prefix}:{abbrev_term}"] = term

    return search_terms
