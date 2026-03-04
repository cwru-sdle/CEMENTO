from copy import deepcopy
from functools import partial

from more_itertools import ilen
from rdflib import Graph, URIRef, OWL, RDF, RDFS

from cemento.rdf.io import aggregate_graphs
from cemento.utils.io import get_data_folders


def is_class(named_individuals, term):
    if term in named_individuals:
        return False
    return isinstance(term, URIRef)

def expand_punned_triples(punned_graph: Graph) -> Graph:
    output_graph = deepcopy(punned_graph)

    data_folder = get_data_folders()
    ref_graph = aggregate_graphs(data_folder)

    eval_graph = punned_graph + ref_graph
    ref_props = set(eval_graph.subjects(RDF.type, RDF.Property))

    # Include subproperties recursively
    all_props = set()
    for p in ref_props:
        all_props |= set(eval_graph.transitive_objects(p, RDFS.subPropertyOf))

    named_individuals = set(eval_graph.subjects(RDF.type, OWL.NamedIndividual))

    # get undefined properties
    undefined_props = set(p for p in punned_graph.predicates() if ilen(ref_graph.objects(p, RDF.type)) <= 0)
    new_obj_props = set()

    is_term_class = partial(is_class, named_individuals)
    for prop in undefined_props:
        s, _, o = next(punned_graph.triples((None, prop, None)), (None, None, None))
        if is_term_class(s) and is_term_class(o):
            new_obj_props.add(prop)

    # add undefined properties to graph
    for prop in new_obj_props:
        output_graph.add((prop, RDF.type, OWL.ObjectProperty))

    # collect domains and ranges for properties in graph
    for prop in undefined_props:
        subj_objs = punned_graph.subject_objects(prop)
        doms, ranges = zip(*subj_objs)
        for dom in doms:
            output_graph.add((prop, RDFS.domain, dom))
        for ran in ranges:
            output_graph.add((prop, RDFS.range, ran))

    return output_graph








