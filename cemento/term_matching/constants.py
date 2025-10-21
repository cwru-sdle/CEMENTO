from enum import Enum

from more_itertools import flatten
from rdflib import DCTERMS, OWL, RDF, RDFS, SKOS, Namespace, URIRef
from thefuzz import fuzz


class FuzzAlgorithm(Enum):
    SimpleRatio = fuzz.ratio
    PartialRatio = fuzz.partial_ratio
    TokenSortRatio = fuzz.token_sort_ratio
    TokenSetRatio = fuzz.token_set_ratio
    PartialTokenSetRatio = fuzz.partial_token_set_ratio


default_namespaces = [RDF, RDFS, OWL, DCTERMS, SKOS]
default_namespace_prefixes = ["rdf", "rdfs", "owl", "dcterms", "skos"]

RANK_PROPS = {RDF.type, RDFS.subClassOf}
FALLBACK_STRAT_TYPES = {
    OWL.bottomDataProperty,
    OWL.topDataProperty,
    OWL.AnnotationProperty,
    OWL.DeprecatedProperty,
    OWL.priorVersion,
    RDF.type,
    OWL.versionInfo,
    OWL.backwardCompatibleWith,
    RDFS.subClassOf,
    OWL.incompatibleWith,
    OWL.DatatypeProperty,
}

SUPPRESSION_KEY = "*"

TRIPLE_SYNTAX_SUGAR = URIRef("https://cwrusdle.bitbucket.io/mds/tripleSyntaxSugar")

valid_collection_types = {
    "owl:unionOf": OWL.unionOf,
    "owl:intersectionOf": OWL.intersectionOf,
    "owl:complementOf": OWL.complementOf,
    "mds:tripleSyntaxSugar": TRIPLE_SYNTAX_SUGAR,
}


def get_default_namespace_prefixes() -> tuple[str, URIRef | Namespace]:
    return {
        prefix: ns
        for prefix, ns in zip(
            default_namespace_prefixes, default_namespaces, strict=True
        )
    }


def get_namespace_terms(*namespaces) -> set[URIRef]:
    namespace_terms = map(dir, namespaces)
    return set(filter(lambda term: isinstance(term, URIRef), flatten(namespace_terms)))
