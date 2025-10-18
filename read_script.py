import re
from enum import Enum
from functools import partial, reduce
from itertools import chain
from operator import itemgetter
from pathlib import Path

from more_itertools import partition, unique_everseen, flatten
from rdflib import RDF, RDFS, SKOS, Graph, Literal, URIRef, XSD
from rdflib.namespace import split_uri
from thefuzz import fuzz
from thefuzz.process import extractOne

from cemento.draw_io.transforms import (
    extract_elements,
    parse_containers,
    parse_elements,
)
from cemento.rdf.transforms import (
    construct_literal,
    get_literal_lang_annotation,
)
from cemento.term_matching.transforms import (
    get_prefixes,
    get_term_search_keys,
)
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
)
from cemento.utils.utils import (
    fst,
    snd,
)


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


def get_namespace_terms(*namespaces) -> set[URIRef]:
    namespace_terms = map(dir, namespaces)
    return set(filter(lambda term: isinstance(term, URIRef), flatten(namespace_terms)))


def substitute_term(
    search_keys: list[str] | str, search_pool: set[tuple[URIRef, str]]
) -> URIRef | None:
    if isinstance(search_keys, str):
        search_keys = [search_keys]
    results = map(
        partial(
            extractOne,
            choices=search_pool,
            processor=lambda item: item[1] if isinstance(item, tuple) else item,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=90,
        ),
        search_keys,
    )
    results = filter(lambda item: item is not None, results)
    best_match = max(results, key=itemgetter(1), default=None)

    if best_match is None:
        return None

    (best_match, _), _ = best_match
    return best_match


def main() -> None:
    input_path = "sandbox/axiom-test.drawio"
    prefixes_file = "cemento/data/default_prefixes.json"
    defaults_folder = "cemento/data/defaults"
    onto_ref_folder = "cemento/data/references"

    prefixes_file = get_default_prefixes_file() if not prefixes_file else prefixes_file
    defaults_folder = (
        get_default_defaults_folder() if not defaults_folder else defaults_folder
    )
    onto_ref_folder = (
        get_default_references_folder() if not onto_ref_folder else onto_ref_folder
    )
    prefixes, inv_prefixes = get_prefixes(prefixes_file, onto_ref_folder)

    elements = parse_elements(input_path)
    containers = parse_containers(elements)
    container_content = set(chain(*containers.values()))
    non_container_elements = {
        key: value for key, value in elements.items() if key not in containers
    }
    term_ids, property_ids = extract_elements(non_container_elements)
    triples = [
        (
            elements[triple_id]["source"],
            triple_id,
            elements[triple_id]["target"],
        )
        for triple_id in property_ids
    ]
    all_terms = unique_everseen(chain(term_ids, property_ids))
    convert_drawio_to_rdf(elements, all_terms, triples, prefixes, inv_prefixes)

def convert_drawio_to_rdf(elements, all_terms, triples, prefixes, inv_prefixes):
    term_dict = {term_id: elements[term_id].get("value", None) for term_id in all_terms}
    uriref_terms, literal_terms = partition(
        lambda item: '"' in item[1], term_dict.items()
    )
    search_pool_files = Path("cemento/data").rglob("*.ttl")
    ref_graph = Graph()
    ref_graph = reduce(lambda acc, item: acc.parse(item), search_pool_files, ref_graph)
    search_pool_terms = chain(
        ref_graph.subjects(predicate=RDFS.subClassOf),
        ref_graph.subjects(predicate=RDF.type),
    )
    search_pool_terms = filter(lambda term: isinstance(term, URIRef), search_pool_terms)
    search_pool_term_prefixes = {
        term: get_uriref_prefix(term, inv_prefixes) for term in search_pool_terms
    }
    search_pool_terms = filter(
        lambda item: item[1] is not None, search_pool_term_prefixes.items()
    )
    search_pool_terms = list(map(fst, search_pool_terms))
    labels = map(
        lambda item: (item, ref_graph.value(subject=item, predicate=RDFS.label)),
        search_pool_terms,
    )
    alt_labels = map(
        lambda item: (item, ref_graph.objects(subject=item, predicate=SKOS.altLabel)),
        search_pool_terms,
    )
    alt_labels = ((key, value) for key, values in alt_labels for value in values)
    abbrev_terms = map(
        lambda item: (item, get_uriref_abbrev_term(item)), search_pool_terms
    )
    ref_search_pool = filter(
        lambda item: item[1] is not None, chain(labels, alt_labels, abbrev_terms)
    )
    ref_search_pool = map(lambda item: (item[0], str(item[1])), ref_search_pool)
    ref_search_pool = map(
        lambda item: (item[0], f"{search_pool_term_prefixes[item[0]]}:{item[1]}"),
        ref_search_pool,
    )
    ref_search_pool = set(ref_search_pool)

    default_prefix = "mds"
    uriref_terms = map(
        lambda item: (
            fst(item),
            f"{default_prefix}:{snd(item)}" if ":" not in snd(item) else snd(item),
        ),
        uriref_terms,
    )

    unlabeled_urirefs, labeled_urirefs = partition(
        lambda item: re.search(r".*\(.*\)", item[1]) is not None, uriref_terms
    )
    labeled_urirefs = list(labeled_urirefs)
    aliases = {
        key: [
            alias.strip()
            for alias in re.match(r".*\((.*)\)", value).group(1).split(",")
            if alias.strip()
        ]
        for key, value in labeled_urirefs
    }

    labeled_urirefs = map(
        lambda item: (item[0], re.match(r"(.*)\(.*\)", item[1]).group(1).strip()),
        labeled_urirefs,
    )
    uriref_terms = chain(unlabeled_urirefs, labeled_urirefs)
    uriref_terms = dict(uriref_terms)

    term_search_keys = map(
        lambda item: (item[0], get_term_search_keys(item[1], inv_prefixes)),
        uriref_terms.items(),
    )
    term_search_keys = dict(term_search_keys)
    term_substitution = {
        key: substitute_term(search_keys, ref_search_pool)
        for key, search_keys in term_search_keys.items()
    }
    substituted, not_substituted = partition(
        lambda item: item[1] is None, term_substitution.items()
    )
    not_substituted, substituted = dict(not_substituted), dict(substituted)
    term_substitution.update(
        {
            key: convert_str_uriref(uriref_terms[key], prefixes)
            for key, value in term_substitution.items()
            if key in not_substituted.keys()
        }
    )

    literal_terms = dict(literal_terms)

    datatype_search_terms = map(
        lambda item: (item, get_uriref_abbrev_term(item)), get_namespace_terms(XSD)
    )
    datatype_search_terms = map(
        lambda item: (item[0], f"xsd:{item[1]}" if item[1] else None),
        datatype_search_terms,
    )
    literal_type_annotations = map(
        lambda item: (item[0], get_datatype_annotation(item[1])), literal_terms.items()
    )
    literal_datatype = map(
        lambda item: (
            item[0],
            substitute_term(item[1], datatype_search_terms),
        ),
        literal_type_annotations,
    )
    literal_datatype = dict(literal_datatype)
    literal_substitution = {
        key: construct_literal(
            literal_str,
            lang=get_literal_lang_annotation(literal_str),
            datatype=literal_datatype[key] or XSD.string,
        )
        for key, literal_str in literal_terms.items()
    }

    term_substitution.update(literal_substitution)

    rdf_graph = Graph()
    for triple in triples:
        triple = tuple(map(lambda item: term_substitution.get(item, None), triple))
        if None in triple:
            continue
        rdf_graph.add(triple)

    ## replace all properties with lowercase equivalents
    defaults_graph = Graph()
    defaults_graph_files = Path("cemento/data/defaults").rglob("*.ttl")
    defaults_graph = reduce(
        lambda acc, item: acc.parse(item), defaults_graph_files, defaults_graph
    )
    property_classes = defaults_graph.transitive_subjects(
        predicate=RDFS.subClassOf, object=RDF.Property
    )
    property_classes = list(property_classes)
    property_triples = rdf_graph.triples_choices((None, RDF.type, property_classes))
    graph_properties = chain(
        map(lambda item: item[0], property_triples), rdf_graph.predicates()
    )
    not_substituted = {term: term_substitution[term] for term in not_substituted}
    graph_properties = set(
        filter(lambda term: term in not_substituted.values(), graph_properties)
    )
    graph_properties = {
        term: value
        for term, value in not_substituted.items()
        if value in graph_properties
    }

    triples_to_add = []
    triples_to_remove = []
    for key, prop in graph_properties.items():
        new_uri = convert_str_uriref(
            convert_uriref_str(prop, inv_prefixes), prefixes, case=TermCase.CAMEL_CASE
        )
        term_substitution[key] = new_uri

        for subj, obj in rdf_graph.subject_objects(predicate=prop):
            triples_to_add.append((subj, new_uri, obj))
            triples_to_remove.append((subj, prop, obj))

        for subj, pred in rdf_graph.subject_predicates(object=prop):
            triples_to_add.append((subj, pred, new_uri))
            triples_to_remove.append((subj, pred, prop))

        for pred, obj in rdf_graph.predicate_objects(subject=prop):
            triples_to_add.append((new_uri, pred, obj))
            triples_to_remove.append((prop, pred, obj))

    for triple in triples_to_remove:
        rdf_graph.remove(triple)

    for triple in triples_to_add:
        rdf_graph.add(triple)

    ## add labels for terms with labels
    for term, aliases in aliases.items():
        label = aliases.pop(0)
        subj = term_substitution[term]
        rdf_graph.add((subj, RDFS.label, Literal(label)))
        for alt_label in aliases:
            rdf_graph.add((subj, SKOS.altLabel, Literal(alt_label)))

    ## import properties for substituted terms
    for _, term in substituted.items():
        imported_triples = get_corresponding_triples(
            ref_graph, term, RDFS.label, RDFS.subClassOf, RDF.type
        )
        for triple in imported_triples:
            rdf_graph.add(triple)
        rdf_graph.add((term, SKOS.exactMatch, term))

    ## remove triples that already deal with default terms
    rdf_graph -= defaults_graph
    for triple in rdf_graph.triples_choices(
        (list(defaults_graph.subjects()), None, None)
    ):
        rdf_graph.remove(triple)

    ## bind prefixes
    for prefix, namespace in prefixes.items():
        rdf_graph.bind(prefix, namespace)

    ## save to file
    rdf_graph.serialize("test.ttl", format="turtle")


if __name__ == "__main__":
    main()
