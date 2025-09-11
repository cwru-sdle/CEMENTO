import re
from collections import defaultdict
from collections.abc import Container, Iterable
from functools import partial, reduce
from itertools import chain
from pathlib import Path

import tldextract
from more_itertools import unique_everseen
from rdflib import OWL, RDF, RDFS, SKOS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import split_uri
from thefuzz import fuzz, process

from cemento.term_matching.constants import (
    FALLBACK_STRAT_TYPES,
    RANK_PROPS,
    get_default_namespace_prefixes,
)
from cemento.term_matching.io import (
    get_rdf_file_iter,
    get_rdf_graph,
    get_search_terms_from_defaults,
    get_search_terms_from_graph,
    read_prefixes_from_graph,
    read_prefixes_from_json,
)
from cemento.term_matching.preprocessing import merge_dictionaries
from cemento.utils.constants import RDFFormat
from cemento.utils.utils import get_abbrev_term, remove_term_names


def search_similar_terms_multikey(
    search_keys: Iterable[str], search_terms: Iterable[str], score_cutoff: int = 80
) -> list[tuple[URIRef, int]]:
    return [
        result
        for search_key in search_keys
        if (
            result := process.extractOne(
                search_key,
                search_terms,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=score_cutoff,
            )
        )
        is not None
    ]


def compare_lower_terms(string1, string2, **kwargs):
    return fuzz.ratio(string1.lower(), string2.lower())


def get_term_matches(
    term: str, search_pool: Container[str], score_cutoff: int = None
) -> tuple[str, int]:
    return process.extractOne(
        term,
        search_pool,
        score_cutoff=score_cutoff,
        scorer=compare_lower_terms,
    )


def substitute_term(
    term: str, search_terms: set[str], score_cutoff=90
) -> tuple[str, bool]:
    matched_rank_term, score = (
        match_res if (match_res := get_term_matches(term, search_terms)) else (None, 0)
    )
    return (
        matched_rank_term if (meets_cutoff := score > score_cutoff) else term,
        meets_cutoff,
    )


def substitute_term_multikey(
    search_keys: Iterable[str],
    search_terms: dict[str, URIRef],
    score_cutoff: int = 80,
    log_results: bool = False,
    suppression_key: str = "*",
) -> URIRef | tuple[URIRef, list[tuple[URIRef, int]]]:

    if any([suppression_key in key for key in search_keys]):
        if log_results:
            return (None, [None], [(None, None)])
        return None

    search_keys = search_keys if not log_results else list(search_keys)
    search_results = search_similar_terms_multikey(
        search_keys, search_terms.keys(), score_cutoff=score_cutoff
    )
    best_match, _ = max(
        search_results,
        key=lambda x: x[1] if x is not None else -1,
        default=(None, -1),
    )
    term_substitute = search_terms[best_match] if best_match else None
    if log_results:
        return (term_substitute, search_keys, search_results)
    return term_substitute


def get_substitute_mapping(
    search_keys: dict[str, list[str]],
    search_pool: dict[str, URIRef],
    terms: Iterable[str],
    log_results: bool = False,
):
    return {
        term: substituted_value
        for term, substituted_value in map(
            lambda term: (
                term,
                substitute_term_multikey(
                    search_keys[term], search_pool, log_results=log_results
                ),
            ),
            unique_everseen(terms),
        )
        if substituted_value is not None
    }


def get_term_search_keys(term: str, inv_prefix: dict[URIRef, str]) -> list[str]:
    prefix, abbrev_term = get_abbrev_term(term)
    undo_camel_case_term = " ".join(
        re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z][a-z]+|[0-9]+", abbrev_term)
    )
    search_keys = [
        remove_term_names(term),
        f"{prefix}:{abbrev_term}",
        f"{prefix}:{undo_camel_case_term}",
    ]
    return [key.strip() for key in search_keys]


def get_term_search_result(
    term: URIRef,
    inv_prefixes: dict[URIRef | Namespace, str],
    search_terms: dict[str, URIRef],
) -> URIRef:
    ns, abbrev_term = split_uri(term)
    prefix = inv_prefixes[str(ns)]
    search_term = f"{prefix}:{abbrev_term}"
    if search_term in search_terms:
        return search_terms[search_term]
    return None


def add_exact_matches(
    term: URIRef, match_properties: dict[URIRef, URIRef | None], rdf_graph: Graph
) -> Graph:
    # if the term is already imported from somewhere else
    # get the type and label if available and add to the ttl file
    for match_property, value in match_properties.items():
        if value:
            rdf_graph.add((term, match_property, value))

    # add an exact match to the ttl file for easier cross-referencing
    rdf_graph.add((term, SKOS.exactMatch, term))
    return rdf_graph


def get_aliases(rdf_graph: Graph) -> dict[URIRef, Literal]:
    label_tuples = list(
        chain(
            rdf_graph.subject_objects(RDFS.label),
            rdf_graph.subject_objects(SKOS.altLabel),
        )
    )
    # TODO: find a more functional approach that works. previous implementation with groupby 'ate' the first term
    aliases = defaultdict(list)
    for key, value in label_tuples:
        aliases[key].append(value)
    return aliases


def get_term_types(rdf_graph: Graph) -> dict[URIRef, URIRef]:
    return {subj: obj for subj, pred, obj in rdf_graph if pred == RDF.type}


def combine_graphs(graphs: Iterable[Graph]) -> Graph:
    return reduce(lambda acc, graph: acc + graph, graphs, Graph())


def generate_residual_prefixes(
    rdf_graph: Graph, inv_prefixes: dict[Namespace | URIRef, str]
) -> dict[str, URIRef | Namespace]:
    new_prefixes = defaultdict(list)
    new_prefix_namespaces = set()
    for subj, pred, obj in rdf_graph:
        for term in [subj, pred, obj]:
            if isinstance(term, URIRef):
                try:
                    ns, abbrev = split_uri(term)
                except ValueError:
                    ns = term
                if ns not in inv_prefixes:
                    new_prefix_namespaces.add(str(ns))
    gns_idx = 0
    for ns in new_prefix_namespaces:
        url_extraction = tldextract.extract(ns)
        new_prefix = res[-1] if (res := re.findall(r"\w+", ns)) else ""
        if url_extraction.suffix and new_prefix in url_extraction.suffix.split("."):
            new_prefix = url_extraction.domain
        new_prefix = re.sub(r"[^a-zA-Z0-9]", "", new_prefix)
        if not new_prefix or new_prefix.isdigit():
            new_prefix = f"gns{gns_idx}"
            gns_idx += 1
        new_prefixes[new_prefix].append(ns)

    return_prefixes = dict()
    for prefix, namespaces in new_prefixes.items():
        if len(namespaces) > 1:
            for idx, ns in enumerate(namespaces):
                return_prefixes[f"{prefix}{idx+1}"] = ns
        else:
            return_prefixes[prefix] = namespaces[0]

    return return_prefixes


def get_prefixes(
    prefixes_path: str | Path,
    onto_ref_folder: str | Path,
    input_file: str | Path = None,
    file_format: RDFFormat | str = None,
) -> tuple[dict[str, URIRef | Namespace], dict[URIRef | Namespace, str]]:
    prefixes = dict()
    if prefixes_path:
        prefixes = read_prefixes_from_json(prefixes_path)

    default_namespace_prefixes = get_default_namespace_prefixes()
    prefixes.update(default_namespace_prefixes)
    inv_prefixes = {value: key for key, value in prefixes.items()}

    if onto_ref_folder:
        file_prefixes = map(
            read_prefixes_from_graph, get_rdf_file_iter(onto_ref_folder)
        )
        prefixes |= merge_dictionaries(file_prefixes)
        inv_prefixes = {value: key for key, value in prefixes.items()}

        residual_file_prefixes = map(
            partial(generate_residual_prefixes, inv_prefixes=inv_prefixes),
            get_rdf_file_iter(onto_ref_folder),
        )
        if input_file is not None:
            residual_input_prefixes = generate_residual_prefixes(
                get_rdf_graph(input_file, file_format),
                inv_prefixes=inv_prefixes,
            )
            residual_file_prefixes = chain(
                residual_file_prefixes, [residual_input_prefixes]
            )
        residual_file_prefixes = {
            key: value
            for residual_prefixes in residual_file_prefixes
            for key, value in residual_prefixes.items()
        }
        prefixes.update(residual_file_prefixes)
        inv_prefixes = {value: key for key, value in prefixes.items()}

    return prefixes, inv_prefixes


def get_default_terms(defaults_folder: str | Path = None):
    default_namespace_prefixes = get_default_namespace_prefixes()
    default_terms_from_lib = {
        term
        for ns in default_namespace_prefixes.values()
        for term in dir(ns)
        if isinstance(term, URIRef)
    }
    if defaults_folder:
        default_terms_from_file = reduce(
            lambda acc, rdf_graph: acc | set(rdf_graph.all_nodes()),
            get_rdf_file_iter(defaults_folder),
            set(),
        )
        default_terms_from_lib |= default_terms_from_file
    default_terms_from_lib = set(
        filter(lambda x: isinstance(x, URIRef), default_terms_from_lib)
    )
    return default_terms_from_lib


def get_search_terms(
    inv_prefixes: dict[URIRef, str],
    onto_ref_folder: str | Path = None,
    defaults_folder: str | Path = None,
    extra_paths: list[str | Path] | None = None,
):
    search_terms = get_search_terms_from_defaults(get_default_namespace_prefixes())

    if defaults_folder:
        defaults_file_search_terms = map(
            partial(get_search_terms_from_graph, inv_prefixes=inv_prefixes),
            get_rdf_file_iter(defaults_folder),
        )
        search_terms |= merge_dictionaries(defaults_file_search_terms)

    if onto_ref_folder:
        file_search_terms = map(
            partial(get_search_terms_from_graph, inv_prefixes=inv_prefixes),
            get_rdf_file_iter(onto_ref_folder),
        )
        search_terms |= merge_dictionaries(file_search_terms)

    # TODO: make more functional using reduce
    if extra_paths is not None:
        for extra_path in extra_paths:
            file_search_terms = map(
                partial(get_search_terms_from_graph, inv_prefixes=inv_prefixes),
                get_rdf_file_iter(extra_path),
            )
            search_terms |= merge_dictionaries(file_search_terms)

    return search_terms


def get_prop_family(rdf_graph: Graph, prop: URIRef) -> set[URIRef]:
    props_from_type = rdf_graph.transitive_subjects(RDF.type, prop)
    props_from_subclass = rdf_graph.transitive_subjects(RDFS.subClassOf, prop)
    return set(chain(props_from_type, props_from_subclass, [prop]))


def get_abbrev_uri(
    default_term: URIRef, inv_prefixes: dict[URIRef | Namespace, str]
) -> str:
    ns, abbrev_term = split_uri(default_term)
    prefix = inv_prefixes[str(ns)]
    return f"{prefix}:{abbrev_term.strip()}"


def get_abbrev_uri_with_prefix(term: URIRef, prefix: str) -> str:
    ns, abbrev_term = split_uri(term)
    return f"{prefix}:{abbrev_term.strip()}"


# TODO: memoize
def get_entire_prop_family(
    defaults_folder: str | Path, inv_prefixes: dict[URIRef | Namespace, str]
) -> set[URIRef]:
    # TODO: move prop_family to constants
    prop_parents = {
        OWL.ObjectProperty,
        OWL.AnnotationProperty,
        OWL.DatatypeProperty,
    }
    return reduce(
        lambda acc, prop: acc
        | set(get_prop_family_from_defaults(prop, defaults_folder, inv_prefixes)),
        prop_parents,
        set(),
    )


def detect_lineage(ref_graph: Graph, term_family: set[URIRef], term: URIRef) -> bool:
    ancestors = list(
        unique_everseen(
            chain(
                ref_graph.transitive_objects(predicate=RDF.type, subject=term),
                ref_graph.transitive_objects(predicate=RDFS.subClassOf, subject=term),
            )
        )
    )
    if any([term != ancestor and ancestor in term_family for ancestor in ancestors]):
        return True
    return False


def get_prop_family_from_defaults(
    prop: URIRef,
    defaults_folder: str | Path,
    inv_prefixes: dict[URIRef | Namespace, str],
) -> Iterable[str]:
    prop_family = map(
        partial(get_prop_family, prop=prop),
        get_rdf_file_iter(defaults_folder),
    )
    prop_family = (prop for props in prop_family for prop in props)
    return prop_family


def get_rank_props() -> Iterable[URIRef]:
    # TODO: add subclass terms to constants
    return RANK_PROPS


def get_strat_props(
    defaults_folder: str | Path,
    inv_prefixes: dict[URIRef | Namespace, str],
    include_non_rank_props: bool = True,
) -> set[str]:
    non_rank_strat_prop_parents = {
        OWL.AnnotationProperty,
        OWL.DatatypeProperty,
    }
    strat_props = get_rank_props()
    if include_non_rank_props:
        try:
            non_rank_strat_props = map(
                partial(
                    get_prop_family_from_defaults,
                    inv_prefixes=inv_prefixes,
                    defaults_folder=defaults_folder,
                ),
                non_rank_strat_prop_parents,
            )
        except Exception:
            non_rank_strat_props = FALLBACK_STRAT_TYPES
        strat_props = chain(strat_props, *non_rank_strat_props)
    return set(strat_props)


def get_abbrev_prefixed_literal(
    term: URIRef, literal: Literal, inv_prefixes: dict[URIRef | Namespace, str]
) -> str:
    ns, _ = split_uri(term)
    prefix = inv_prefixes[str(ns)]
    return f"{prefix}:{literal.lower().strip()}"


def get_preds_in_ref(rdf_graph: Graph, type_refs: set[URIRef]) -> Iterable[URIRef]:
    return (
        subj for subj, obj in rdf_graph.subject_objects(RDF.type) if obj in type_refs
    )


def get_term_aliases_from_graph(rdf_graph: Graph, term: URIRef) -> list[URIRef]:
    return (
        term,
        chain(
            rdf_graph.objects(term, RDFS.label),
            rdf_graph.objects(term, SKOS.altLabel),
        ),
    )


# TODO: make these functions more pure by taking rdf_graphs and type_refs
def get_strat_predicates(
    onto_ref_folder: str | Path,
    defaults_folder: str | Path,
    inv_prefixes: dict[URIRef | Namespace, str],
) -> list[URIRef]:
    type_refs = get_strat_props(defaults_folder, inv_prefixes)
    return list(
        chain(
            *map(
                partial(get_preds_in_ref, type_refs=type_refs),
                get_rdf_file_iter(onto_ref_folder),
            )
        )
    )


def get_strat_predicates_str(
    onto_ref_folder: str | Path,
    defaults_folder: str | Path,
    inv_prefixes: dict[URIRef | Namespace, str],
) -> set[str]:
    strat_preds = get_strat_predicates(onto_ref_folder, defaults_folder, inv_prefixes)
    stat_pred_aliases = (
        get_term_aliases_from_graph(graph, pred)
        for graph in get_rdf_file_iter(onto_ref_folder)
        for pred in strat_preds
    )
    stat_preds_str = map(
        partial(get_abbrev_uri, inv_prefixes=inv_prefixes), strat_preds
    )
    aliased_stat_preds_str = (
        get_abbrev_prefixed_literal(term, alias, inv_prefixes)
        for term, aliases in stat_pred_aliases
        for alias in aliases
    )
    rank_props_str = map(
        partial(get_abbrev_uri, inv_prefixes=inv_prefixes), get_rank_props()
    )
    return set(chain(stat_preds_str, aliased_stat_preds_str, rank_props_str))
