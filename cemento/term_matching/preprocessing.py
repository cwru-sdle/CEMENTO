import re
from enum import Enum

from rdflib import URIRef, Graph
from rdflib.namespace import split_uri

from cemento.term_matching.constants import SUPPRESSION_KEY


def merge_dictionaries(dict_list: list[dict[any, any]]) -> dict[any, any]:
    return {key: value for each_dict in dict_list for key, value in each_dict.items()}


def remove_suppression_key(term: str, suppression_key=SUPPRESSION_KEY) -> str:
    return term.replace(suppression_key, "")

def remove_facet_info(term: str) -> str:
    return re.sub(r'\[.*\]',"", term).strip()



class TermCase(Enum):
    PASCAL_CASE = "pascal"
    CAMEL_CASE = "camel_case"


def convert_uriref_str(term: URIRef, inv_prefixes: tuple[str, str]):
    ns, abbrev_term = split_uri(term)
    prefix = inv_prefixes[ns]
    return f"{prefix}:{abbrev_term}"


def get_uriref_abbrev_term(term: URIRef) -> str:
    _, abbrev_term = split_uri(term)
    return abbrev_term


def get_datatype_annotation(literal_str: str) -> str:
    datatype = res[0] if (res := re.findall(r"\^\^(\w+:\w+)", literal_str)) else None
    return datatype


def get_uriref_prefix(term: URIRef, inv_prefixes: dict[str, str]) -> str | None:
    try:
        ns, _ = split_uri(term)
    except ValueError:
        return None
    return inv_prefixes.get(ns, None)


def get_character_words(term: str):
    words = re.sub(r"[^a-zA-Z0-9]+", " ", term)
    return words.split(" ")


def convert_to_pascal_case(term: str) -> str:
    words = get_character_words(term)
    words = [
        (word[0].upper() + (word[1:] if len(word) > 1 else "")).strip()
        for word in words
    ]
    return "".join(words)


def convert_to_camel_case(term: str) -> str:
    pascal_case = convert_to_pascal_case(term)
    return pascal_case[0].lower() + (pascal_case[1:] if len(pascal_case) > 1 else "")


def get_corresponding_triples(ref_graph: Graph, term: URIRef, *predicates: URIRef):
    return [
        (term, pred, val)
        for pred in predicates
        if (val := ref_graph.value(subject=term, predicate=pred)) is not None
    ]


def convert_str_uriref(
    term: str, prefixes: tuple[str, str], case: TermCase = TermCase.PASCAL_CASE
):
    prefix, abbrev_term = term.split(":")
    if case == TermCase.CAMEL_CASE:
        abbrev_term = convert_to_camel_case(abbrev_term)
    else:
        abbrev_term = convert_to_pascal_case(abbrev_term)
    ns = prefixes[prefix]
    return URIRef(f"{ns}{abbrev_term}")
