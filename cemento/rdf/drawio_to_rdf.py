import re
from functools import reduce
from itertools import chain
from pathlib import Path

from more_itertools import partition
from rdflib import RDF, RDFS, Graph, Literal
from rdflib import SKOS, XSD

from cemento.draw_io.read_diagram import read_drawio
from cemento.rdf.io import aggregate_graphs
from cemento.rdf.preprocessing import extract_aliases
from cemento.rdf.transforms import (
    construct_literal,
    get_literal_lang_annotation,
    get_term_search_pool,
)
from cemento.term_matching.constants import get_namespace_terms
from cemento.term_matching.preprocessing import (
    get_uriref_abbrev_term,
    convert_str_uriref,
    get_datatype_annotation,
    convert_uriref_str,
    TermCase,
    get_corresponding_triples,
)
from cemento.term_matching.transforms import (
    get_prefixes,
    get_term_search_keys,
    substitute_term,
)
from cemento.utils.constants import RDFFormat
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
    get_rdf_format,
)
from cemento.utils.utils import fst, snd


def convert_drawio_to_rdf(
    input_path: str | Path,
    output_path: str | Path,
    file_format: str | RDFFormat = None,
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    prefixes_path: str | Path = None,
    check_errors: bool = False,
    log_substitution_path: str | Path = None,
) -> None:
    elements, all_terms, triples = read_drawio(
        input_path,
        check_errors=check_errors,
    )
    convert_graph_to_rdf_file(
        elements,
        all_terms,
        triples,
        output_path,
        file_format=file_format,
        onto_ref_folder=onto_ref_folder,
        defaults_folder=defaults_folder,
        prefixes_path=prefixes_path,
        log_substitution_path=log_substitution_path,
    )


def convert_graph_to_rdf_graph(
    elements,
    all_terms,
    triples,
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    prefixes_path: str | Path = None,
    log_substitution_path: str | Path = None,
) -> Graph:
    onto_ref_folder = (
        get_default_references_folder() if not onto_ref_folder else onto_ref_folder
    )
    defaults_folder = (
        get_default_defaults_folder() if not defaults_folder else defaults_folder
    )
    prefixes_path = get_default_prefixes_file() if not prefixes_path else prefixes_path
    prefixes, inv_prefixes = get_prefixes(prefixes_path, onto_ref_folder)

    term_dict = {term_id: elements[term_id].get("value", None) for term_id in all_terms}
    uriref_terms, literal_terms = partition(
        lambda item: '"' in item[1], term_dict.items()
    )
    ref_graph = aggregate_graphs(onto_ref_folder)
    ref_search_pool = get_term_search_pool(ref_graph, inv_prefixes)

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
    aliases = dict(
        map(lambda item: (item[0], extract_aliases(item[1])), labeled_urirefs)
    )
    cleaned_labeled_urirefs = map(
        lambda item: (item[0], re.match(r"(.*)\(.*\)", item[1]).group(1).strip()),
        labeled_urirefs,
    )
    uriref_terms = chain(unlabeled_urirefs, cleaned_labeled_urirefs)
    uriref_terms = dict(uriref_terms)

    term_search_keys = map(
        lambda item: (item[0], get_term_search_keys(item[1], inv_prefixes)),
        uriref_terms.items(),
    )
    term_search_keys = dict(term_search_keys)
    term_substitution = {
        key: substitute_term(search_keys, set(ref_search_pool))
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
    datatype_search_terms = dict(datatype_search_terms)
    datatype_search_terms.update(ref_search_pool)
    literal_type_annotations = map(
        lambda item: (item[0], get_datatype_annotation(item[1])), literal_terms.items()
    )
    literal_datatype = map(
        lambda item: (
            item[0],
            substitute_term(item[1], set(datatype_search_terms.items())),
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

    return rdf_graph


def convert_graph_to_rdf_file(
    elements,
    all_terms,
    triples,
    output_path: str | Path,
    file_format: str | RDFFormat = None,
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    prefixes_path: str | Path = None,
    log_substitution_path: str | Path = None,
):
    rdf_format = get_rdf_format(output_path, file_format=file_format)
    rdf_graph = convert_graph_to_rdf_graph(
        elements,
        all_terms,
        triples,
        onto_ref_folder=onto_ref_folder,
        defaults_folder=defaults_folder,
        prefixes_path=prefixes_path,
        log_substitution_path=log_substitution_path,
    )
    rdf_graph.serialize(destination=output_path, format=rdf_format)
