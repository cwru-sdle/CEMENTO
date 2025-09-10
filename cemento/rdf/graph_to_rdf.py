from collections import defaultdict
from copy import deepcopy
from functools import partial, reduce
from itertools import chain, filterfalse, starmap
from pathlib import Path

import networkx as nx
import rdflib
from more_itertools import unique_everseen
from networkx import DiGraph
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

from cemento.rdf.filters import term_in_search_results, term_not_in_default_namespace
from cemento.rdf.io import (
    get_diagram_terms_iter,
    get_diagram_terms_iter_with_pred,
    get_properties_in_file,
    save_substitute_log,
)
from cemento.rdf.preprocessing import (
    get_term_aliases,
)
from cemento.rdf.transforms import (
    add_collection_links_to_graph,
    add_domains_ranges,
    add_labels,
    add_rdf_triples,
    bind_prefixes,
    construct_literal,
    construct_term_uri,
    get_class_terms,
    get_collection_in_edges,
    get_collection_nodes,
    get_collection_subgraph,
    get_collection_triples_and_targets,
    get_domains_ranges,
    get_literal_data_type,
    get_literal_lang_annotation,
    get_xsd_terms,
    remove_generic_property,
)
from cemento.term_matching.constants import get_default_namespace_prefixes
from cemento.term_matching.io import get_rdf_file_iter
from cemento.term_matching.transforms import (
    add_exact_matches,
    combine_graphs,
    get_prefixes,
    get_search_terms,
    get_substitute_mapping,
    get_term_search_keys,
    get_term_types,
)
from cemento.utils.constants import NullTermError, RDFFormat, valid_collection_types
from cemento.utils.io import (
    get_default_defaults_folder,
    get_default_prefixes_file,
    get_default_references_folder,
    get_rdf_format,
)
from cemento.utils.utils import (
    chain_filter,
    enforce_camel_case,
    fst,
    get_abbrev_term,
    snd,
)


def convert_graph_to_rdf_graph(
        graph: DiGraph,
        collect_domains_ranges: bool = False,
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
    search_terms = get_search_terms(inv_prefixes, onto_ref_folder, defaults_folder)

    # TODO: reference from constants file once moved
    # TODO: replace with proper-cased terms once substitute issue is resolved
    collection_nodes = get_collection_nodes(graph)
    collection_subgraph = get_collection_subgraph(graph, collection_nodes)
    collection_in_edges = get_collection_in_edges(collection_nodes.keys(), graph)
    collection_in_edge_labels = list(
        map(
            lambda x: enforce_camel_case(x[2]["label"]),
            filter(lambda x: "label" in x[2], collection_in_edges),
        )
    )
    collection_in_edge_labels_iter = map(
        lambda x: (enforce_camel_case(x), True), collection_in_edge_labels
    )
    nodes_to_remove = set(collection_nodes.keys()) | valid_collection_types
    collection_subgraph = get_collection_subgraph(set(collection_nodes.keys()), graph)
    graph.remove_nodes_from(nodes_to_remove)

    aliases = {
        term: aliases
        for term, aliases in map(
            lambda term: (term, get_term_aliases(term)),
            unique_everseen(
                chain(get_diagram_terms_iter(graph), collection_in_edge_labels)
            ),
        )
    }

    # TODO: assign literal terms IDs so identical values get treated separately
    literal_terms = {
        term
        for term in filter(
            lambda term: ('"' in term),
            unique_everseen(
                chain(get_diagram_terms_iter(graph), collection_in_edge_labels)
            ),
        )
    }

    # process search keys now for partial substitution and full substitution later on
    search_keys = {
        term: search_key
        for term, search_key in map(
            lambda term: (term, get_term_search_keys(term, inv_prefixes)),
            unique_everseen(
                chain(get_diagram_terms_iter(graph), collection_in_edge_labels)
            ),
        )
    }

    # retrieve list of property terms in file to enforce camel case appropriately
    # TODO: determine if user can toggle camel case enforcement for their term
    property_terms = get_properties_in_file(
        search_keys,
        chain(get_diagram_terms_iter(graph), collection_in_edge_labels),
        graph,
        defaults_folder,
        inv_prefixes,
    )

    # get the list of terms from which to consruct URIRefs and Literals and create them
    construct_term_inputs = list(
        filter(
            lambda x: fst(x) not in literal_terms,
            chain(
                get_diagram_terms_iter_with_pred(graph, property_terms),
                collection_in_edge_labels_iter,
            ),
        )
    )
    try:
        constructed_terms = {
            term: term_uri_ref
            for term, term_uri_ref in map(
                lambda term_info: (
                    fst(term_info),
                    construct_term_uri(
                        *get_abbrev_term(fst(term_info), is_predicate=snd(term_info)),
                        prefixes=prefixes,
                    ),
                ),
                construct_term_inputs,
            )
        }
    except KeyError as e:
        offending_key = e.args[0]
        if prefixes_path:
            if Path(prefixes_path) == get_default_prefixes_file():
                raise ValueError(
                    f"The prefix {offending_key} was used but it was not in the default_prefixes.json file. Please consider making your own file and adding it there. Don't forget to set '--prefixes-file-path' when using the cli or setting 'prefixes_path' arguments when scripting."
                ) from KeyError
            else:
                raise ValueError(
                    f"The prefix {offending_key} was used but it was not in the prefix.json file located in {prefixes_path}. Please consider adding it there."
                ) from KeyError
        else:
            raise ValueError(
                f"The prefix {offending_key} was used but it is not part of the default namespace. Consider creating a prefixes.json file and add set the prefixes_path argument."
            ) from KeyError
    except NullTermError:
        raise NullTermError(
            "A null term has been detected. Please make sure all your arrows and shapes are labelled properly."
        ) from NullTermError

    substitution_results = get_substitute_mapping(
        search_keys,
        search_terms,
        chain(get_diagram_terms_iter(graph), collection_in_edge_labels),
        log_results=bool(log_substitution_path),
    )

    if log_substitution_path:
        save_substitute_log(substitution_results, log_substitution_path)
        substitution_results = {
            key: matched_term
            for key, (
                matched_term,
                _,
                _,
            ) in substitution_results.items()
            if matched_term is not None
        }

    preferred_alias_keyed_inv_constructed_terms = dict()
    for key, value in constructed_terms.items():
        compare_value = preferred_alias_keyed_inv_constructed_terms.get(value, "")
        preferred_alias_keyed_inv_constructed_terms[value] = max(
            compare_value, key, key=len
        )

    constructed_terms.update(substitution_results)

    # get datatypes in graph first
    datatype_search_terms = get_xsd_terms()
    datatype_search_terms.update(search_terms)
    constructed_literal_terms = {
        term: construct_literal(
            term,
            lang=get_literal_lang_annotation(term),
            datatype=get_literal_data_type(term, datatype_search_terms),
        )
        for term in literal_terms
    }
    constructed_terms.update(constructed_literal_terms)

    # # create the rdf graph to store the ttl output
    rdf_graph = rdflib.Graph()

    # create the output graph to store valid graph triples
    output_graph = nx.DiGraph()

    # create rdf triples from the collections
    # sort the collection via dfs postorder to start with innermost collection
    collection_triples, collection_targets = get_collection_triples_and_targets(
        collection_nodes, collection_subgraph, rdf_graph, constructed_terms
    )

    for triple in collection_triples:
        rdf_graph.add(triple)

    graph = add_collection_links_to_graph(
        collection_in_edges, collection_targets, graph
    )

    # filter for valid triples and add to output graph
    for subj, obj, data in graph.edges(data=True):
        pred = data["label"]
        # do final null check on triples to add
        if not all((term for term in (subj, obj, pred))):
            print(
                f"[WARNING] the triple ({subj}, {pred}, {obj}) had null values that passed through diagram checks. Not adding to the graph..."
            )
            continue
        subj, obj, pred = tuple(
            constructed_terms[key] if type(key) not in {URIRef, Literal, BNode} else key
            for key in (subj, obj, pred)
        )
        output_graph.add_edge(subj, obj, label=pred)
    class_terms = get_class_terms(output_graph)
    predicate_terms = {data["label"] for _, _, data in output_graph.edges(data=True)}
    literal_terms = set(constructed_literal_terms.values())
    class_terms -= predicate_terms
    output_graph_nodes = set(
        filter(lambda x: not isinstance(x, BNode), output_graph.nodes)
    )
    all_terms = (output_graph_nodes | predicate_terms) - literal_terms

    # bind prefixes to namespaces for the rdf graph
    rdf_graph = bind_prefixes(rdf_graph, prefixes)

    # add all of the class terms as a type
    rdf_graph = add_rdf_triples(
        rdf_graph, ((term, RDF.type, OWL.Class) for term in class_terms)
    )

    # if the term is a predicate and is not part of the default namespaces, add an object property type to the ttl file
    ref_graph = deepcopy(rdf_graph)
    if onto_ref_folder:
        ref_graph += combine_graphs(get_rdf_file_iter(onto_ref_folder))
    term_types = get_term_types(ref_graph)
    term_not_in_default_namespace_filter = partial(
        term_not_in_default_namespace,
        inv_prefixes=inv_prefixes,
        default_namespace_prefixes=get_default_namespace_prefixes(),
    )
    term_type_subs = {
        key: value
        for key, value in map(
            lambda term: (
                term,
                # Assume a custom property is just an Object Property if term type undetermined
                term_types[term] if term in term_types else OWL.ObjectProperty,
            ),
            filter(term_not_in_default_namespace_filter, predicate_terms),
        )
    }
    rdf_graph = add_rdf_triples(
        rdf_graph,
        (
            (term, RDF.type, term_type_subs[term])
            for term in filter(term_not_in_default_namespace_filter, predicate_terms)
        ),
    )

    term_in_search_results_filter = partial(
        term_in_search_results, inv_prefixes=inv_prefixes, search_terms=search_terms
    )

    if onto_ref_folder:
        exact_match_property_predicates = [RDF.type, RDFS.label]
        exact_match_candidates = list(
            chain_filter(
                all_terms,
                term_in_search_results_filter,
                term_not_in_default_namespace_filter,
            )
        )
        exact_match_property_tuples = {
            (term, prop, value)
            for term in exact_match_candidates
            for prop in exact_match_property_predicates
            for value in list(ref_graph.objects(term, prop))
        }
        exact_match_properties = defaultdict(dict)
        for key, prop, value in exact_match_property_tuples:
            exact_match_properties[key][prop] = value

        rdf_graph = reduce(
            lambda rdf_graph, graph_term: add_exact_matches(
                term=graph_term,
                match_properties=exact_match_properties[graph_term],
                rdf_graph=rdf_graph,
            ),
            exact_match_candidates,
            rdf_graph,
        )

    rdf_graph = reduce(
        lambda rdf_graph, graph_term: add_labels(
            term=graph_term,
            labels=aliases[preferred_alias_keyed_inv_constructed_terms[graph_term]],
            rdf_graph=rdf_graph,
        ),
        filter(
            term_not_in_default_namespace_filter,
            filterfalse(term_in_search_results_filter, all_terms),
        ),
        rdf_graph,
    )

    if collect_domains_ranges:
        predicate_domains_ranges = map(
            partial(get_domains_ranges, graph=output_graph),
            filter(
                term_not_in_default_namespace_filter,
                filterfalse(term_in_search_results_filter, predicate_terms),
            ),
        )
        rdf_graph = reduce(
            lambda rdf_graph, triples: add_domains_ranges(triples, rdf_graph),
            predicate_domains_ranges,
            rdf_graph,
        )

    # now add the triples from the drawio diagram
    for domain_term, range_term, data in output_graph.edges(data=True):
        predicate_term = data["label"]
        rdf_graph.add((domain_term, predicate_term, range_term))

    # replace predicate types if another type than owl:ObjectProperty is defined
    rdf_graph = remove_generic_property(rdf_graph, default_property=OWL.ObjectProperty)

    # remove terms that are already in the default namespace if they are subjects
    default_terms = list(filterfalse(term_not_in_default_namespace_filter, all_terms))
    redundant_default_triples = rdf_graph.triples_choices((default_terms, None, None))
    rdf_graph = reduce(
        lambda rdf_graph, triple: rdf_graph.remove(triple),
        redundant_default_triples,
        rdf_graph,
    )

    return rdf_graph


def convert_graph_to_rdf_file(
        element_graph: DiGraph,
        restriction_graph: DiGraph,
        output_path: str | Path,
        file_format: str | RDFFormat = None,
        collect_domains_ranges: bool = False,
        onto_ref_folder: str | Path = None,
        defaults_folder: str | Path = None,
        prefixes_path: str | Path = None,
        log_substitution_path: str | Path = None,
):
    rdf_format = get_rdf_format(output_path, file_format=file_format)
    convert_to_rdf_graph = partial(
        convert_graph_to_rdf_graph,
        onto_ref_folder=onto_ref_folder,
        defaults_folder=defaults_folder,
        prefixes_path=prefixes_path,
        log_substitution_path=log_substitution_path,
    )
    # element_rdf_graph = convert_to_rdf_graph(element_graph)
    restriction_rdf_graph = convert_to_rdf_graph(restriction_graph)
    restriction_rdf_graph.serialize(destination=output_path, format=rdf_format)
