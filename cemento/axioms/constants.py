from rdflib import URIRef, SKOS, XSD

from cemento.rdf.io import aggregate_graphs
from cemento.utils.io import get_default_reserved_folder

symbol_mapping = {
    ">=": XSD.maxInclusive,
    ">": XSD.maxExclusive,
    "<=": XSD.minInclusive,
    "<": XSD.minExclusive,
    "minLength": XSD.minLength,
    "maxLength": XSD.maxLength,
    "length": XSD.length,
}


def get_ms_turtle_mapping() -> dict[URIRef, URIRef]:
    reserved_folder = get_default_reserved_folder()
    reserved_graph = aggregate_graphs(reserved_folder)
    exact_match_triples = reserved_graph.subject_objects(predicate=SKOS.exactMatch)
    return {subj: obj for subj, obj in exact_match_triples}
