import re
from collections.abc import Iterable
from functools import reduce, partial
from itertools import chain
from pathlib import Path

import networkx as nx
from more_itertools import partition
from more_itertools.recipes import flatten
from networkx import DiGraph
from rdflib import RDF, RDFS, Graph, Literal, BNode, OWL
from rdflib import SKOS, XSD
from rdflib.collection import Collection

from cemento.axioms.extract_axioms import extract_axiom_graph
from cemento.axioms.modules import MS
from cemento.draw_io.read_diagram import read_drawio
from cemento.rdf.io import aggregate_graphs, save_substitution_log
from cemento.rdf.preprocessing import extract_aliases
from cemento.rdf.transforms import (
    construct_literal,
    get_literal_lang_annotation,
    get_term_search_pool,
    replace_term_in_triples,
    get_classes_instances,
    get_child_type,
)
from cemento.term_matching.constants import (
    get_namespace_terms,
    SUPPRESSION_KEY,
    valid_collection_types,
    TRIPLE_SYNTAX_SUGAR,
)
from cemento.term_matching.preprocessing import (
    get_uriref_abbrev_term,
    convert_str_uriref,
    get_datatype_annotation,
    convert_uriref_str,
    TermCase,
    get_corresponding_triples,
    remove_suppression_key,
    remove_facet_info,
)
from cemento.term_matching.transforms import (
    get_prefixes,
    get_term_search_keys,
    substitute_term,
)
from cemento.utils.constants import RDFFormat, invert_tuple
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
    get_rdf_format,
    get_default_reserved_folder,
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
    elements, all_terms, triples, containers = read_drawio(
        input_path,
        check_errors=check_errors,
    )
    rdf_format = get_rdf_format(output_path, file_format=file_format)
    rdf_graph = convert_graph_to_rdf_graph(
        elements,
        all_terms,
        triples,
        containers,
        onto_ref_folder=onto_ref_folder,
        defaults_folder=defaults_folder,
        prefixes_path=prefixes_path,
        log_substitution_path=log_substitution_path,
    )
    rdf_graph.serialize(destination=output_path, format=rdf_format)


def convert_graph_to_rdf_graph(
    elements: dict[str, dict[str, any]],
    all_terms: Iterable[str],
    triples: list[tuple[str, str, str]],
    containers: list[tuple[str, str, list[str]]],
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    prefixes_path: str | Path = None,
    log_substitution_path: str | Path = None,
    enforce_camel_case: bool = True,
) -> Graph:
    onto_ref_folder = (
        get_default_references_folder() if not onto_ref_folder else onto_ref_folder
    )
    defaults_folder = (
        get_default_defaults_folder() if not defaults_folder else defaults_folder
    )
    reserved_folder = get_default_reserved_folder()
    log_substitution_path = (
        Path(log_substitution_path) if log_substitution_path else None
    )
    prefixes_path = get_default_prefixes_file() if not prefixes_path else prefixes_path
    prefixes, inv_prefixes = get_prefixes(prefixes_path, onto_ref_folder)

    term_dict = {term_id: elements[term_id].get("value", None) for term_id in all_terms}
    uriref_terms, literal_terms = partition(
        lambda item: '"' in item[1], term_dict.items()
    )
    ref_graph = aggregate_graphs(onto_ref_folder)
    defaults_graph = aggregate_graphs(defaults_folder)
    reserved_graph = aggregate_graphs(reserved_folder)
    ref_graph += defaults_graph + reserved_graph
    ref_search_pool = get_term_search_pool(ref_graph, inv_prefixes)
    datatype_search_terms = map(
        lambda item: (item, get_uriref_abbrev_term(item)), get_namespace_terms(XSD)
    )
    datatype_search_terms = map(
        lambda item: (item[0], f"xsd:{item[1]}" if item[1] else None),
        datatype_search_terms,
    )
    datatype_search_terms = dict(datatype_search_terms)
    datatype_search_terms.update(ref_search_pool)

    ref_search_pool.update(datatype_search_terms.items())

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

    ## exclude terms with suppression key in term_search_keys
    exclude_term = dict(
        filter(lambda item: SUPPRESSION_KEY in item[1], uriref_terms.items())
    )

    term_search_keys = map(
        lambda item: (item[0], get_term_search_keys(item[1], inv_prefixes)),
        uriref_terms.items(),
    )
    term_search_keys = dict(term_search_keys)
    term_substitution = {
        key: (
            None
            if key in exclude_term
            else substitute_term(search_keys, set(ref_search_pool))
        )
        for key, search_keys in term_search_keys.items()
    }
    substituted, not_substituted = partition(
        lambda item: item[1] is None, term_substitution.items()
    )
    not_substituted, substituted = dict(not_substituted), dict(substituted)
    cleaning_steps = [remove_suppression_key, remove_facet_info]
    clean_term = partial(
        lambda item: reduce(lambda val, func: func(val), cleaning_steps, item)
    )
    term_substitution.update(
        {
            key: convert_str_uriref(clean_term(uriref_terms[key]), prefixes)
            for key, value in term_substitution.items()
            if key in not_substituted.keys()
        }
    )

    if log_substitution_path is not None:
        save_substitution_log(
            uriref_terms,
            term_search_keys,
            term_substitution,
            substituted.keys(),
            log_substitution_path,
        )

    literal_terms = dict(literal_terms)

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
            clean_term(literal_str),
            lang=get_literal_lang_annotation(literal_str),
            datatype=literal_datatype[key],
        )
        for key, literal_str in literal_terms.items()
    }

    term_substitution.update(literal_substitution)

    ## substitute containers with proper BNode
    collection_headers = {key: BNode() for key in containers}
    term_substitution.update(collection_headers)

    ## find axiom subgraphs and exclude triples from rdf for now
    term_graph = DiGraph()
    for subj, pred, obj in triples:
        term_graph.add_edge(subj, obj, label=pred)

    ### locate the axiom subgraph triples
    restriction_nodes = filter(
        lambda item: item[1] == OWL.Restriction, term_substitution.items()
    )
    restriction_nodes = list(map(fst, restriction_nodes))
    restriction_triples = map(
        lambda head_node: term_graph.subgraph(
            nx.descendants(term_graph, head_node) | {head_node}
        )
        .copy()
        .edges,
        restriction_nodes,
    )
    restriction_triples = set(flatten(restriction_triples))
    intro_restriction_triples = list(
        filter(
            lambda triple: triple[1] in restriction_nodes, term_graph.edges(data=True)
        )
    )
    restriction_triples |= set(
        map(lambda item: (item[0], item[1]), intro_restriction_triples)
    )
    triples = filter(
        lambda triple: (triple[0], triple[2]) not in restriction_triples, triples
    )

    rdf_graph = Graph()
    for triple in triples:
        triple = tuple(map(lambda item: term_substitution.get(item, None), triple))
        if None in triple:
            continue
        rdf_graph.add(triple)

    ## get classes and instances to classify collection members and annotate later
    classes, instances = get_classes_instances(rdf_graph)

    # add the collections to the graph
    for key, (label, children) in containers.items():
        label = (
            TRIPLE_SYNTAX_SUGAR
            if not label.strip()
            else substitute_term(
                label, set(invert_tuple(valid_collection_types.items()))
            )
        )

        if label is None:
            raise ValueError(
                f"The chosen label is not a valid collection type. Please choose from: {valid_collection_types.keys()}"
            )

        children = [term_substitution[item] for item in children]

        if label == TRIPLE_SYNTAX_SUGAR:
            collection_triples = rdf_graph.triples(
                (None, None, collection_headers[key])
            )
            remove_triples = []
            for subj, pred, obj in collection_triples:
                for child in children:
                    rdf_graph.add((subj, pred, child))
                remove_triples.append((subj, pred, obj))
            for triple in remove_triples:
                rdf_graph.remove(triple)
            del collection_headers[key]
        else:
            first_child_type = get_child_type(classes, instances, children[0])
            if any(
                (child_type := get_child_type(classes, instances, term))
                != first_child_type
                and child_type != OWL.Nothing
                for term in children
            ):
                child_types = {
                    term: get_child_type(classes, instances, term) for term in children
                }
                raise ValueError(
                    f"Cannot combine terms with different types in a collection. Combinations should only be BNodes, OWL.Class or OWL.NamedIndividual, but BNodes (OWL.Nothing) can be mixed with any type. The terms provided have the associated types {child_types}."
                )
            collection_bnode = BNode()
            Collection(rdf_graph, collection_bnode, children)
            if first_child_type is not None:
                rdf_graph.add((collection_headers[key], RDF.type, first_child_type))
            rdf_graph.add((collection_headers[key], label, collection_bnode))

    ## expand axiom terms
    if len(restriction_nodes) > 0:
        ### first extract faceted terms to process
        faceted_terms = filter(
            lambda item: re.search(r".*\[.*\]", item[1]),
            chain(uriref_terms.items(), literal_terms.items()),
        )
        faceted_terms = dict(faceted_terms)
        axiom_graph = extract_axiom_graph(
            term_graph,
            term_substitution,
            restriction_nodes,
            collection_headers,
            intro_restriction_triples,
            faceted_terms,
        )
        rdf_graph += axiom_graph

    ## retrieve all defined properties for annotation logic
    property_classes = defaults_graph.transitive_subjects(
        predicate=RDFS.subClassOf, object=RDF.Property
    )
    property_classes = list(property_classes)
    property_triples = list(rdf_graph.triples_choices((None, None, property_classes)))
    property_object_preds = [MS.onProperty, OWL.onProperty]
    property_objects = list(
        rdf_graph.triples_choices((None, property_object_preds, None))
    )
    graph_properties = chain(
        map(lambda item: item[0], property_triples),
        map(lambda item: item[2], property_objects),
        rdf_graph.predicates(),
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
    graph_prop_urirefs = set(graph_properties.values())

    ## remove properties from classes and instances when annotating types
    classes -= graph_prop_urirefs
    instances -= graph_prop_urirefs

    ## replace all properties with lowercase equivalents
    if enforce_camel_case:
        prop_rename_dict = {
            key: (
                prop
                if key in exclude_term
                else convert_str_uriref(
                    convert_uriref_str(prop, inv_prefixes),
                    prefixes,
                    case=TermCase.CAMEL_CASE,
                )
            )
            for key, prop in graph_properties.items()
        }
        term_substitution.update(prop_rename_dict)
        prop_updated_iri_dict = {
            prop: prop_rename_dict[key] for key, prop in graph_properties.items()
        }
        for prop, new_iri in prop_updated_iri_dict.items():
            rdf_graph = replace_term_in_triples(rdf_graph, prop, new_iri)

    ## add term types to the graph
    for term in classes:
        rdf_graph.add((term, RDF.type, OWL.Class))
    for term in instances:
        rdf_graph.add((term, RDF.type, OWL.NamedIndividual))

    ## add labels for terms with labels
    for term, aliases in aliases.items():
        label = aliases.pop(0)
        subj = term_substitution[term]
        rdf_graph.add((subj, RDFS.label, Literal(label)))
        for alt_label in aliases:
            rdf_graph.add((subj, SKOS.altLabel, Literal(alt_label)))

    ## import properties for substituted terms
    ### exempt reserved keyword terms from imports
    import_terms = filter(
        lambda item: item[1] not in reserved_graph.all_nodes(), substituted.items()
    )
    import_terms = map(snd, import_terms)
    for term in import_terms:
        imported_triples = get_corresponding_triples(
            ref_graph, term, RDFS.label, RDF.type, RDFS.domain, RDFS.range
        )
        for triple in imported_triples:
            rdf_graph.add(triple)
        rdf_graph.add((term, SKOS.exactMatch, term))

    # remove triples that already deal with default terms
    rdf_graph -= defaults_graph
    for triple in rdf_graph.triples_choices(
        (list(defaults_graph.subjects()), None, None)
    ):
        rdf_graph.remove(triple)

    ## bind prefixes
    for prefix, namespace in prefixes.items():
        rdf_graph.bind(prefix, namespace)

    return rdf_graph
