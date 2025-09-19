from rdflib import RDF, RDFS

from cemento.axioms.modules import MS

combinators = {MS.And, MS.Or}

prop_rest_preds = {MS.max, MS.only, MS.that, MS.exactly, MS.value}

class_rest_preds = {MS.equivalentTo, RDF.type, RDFS.subClassOf}
